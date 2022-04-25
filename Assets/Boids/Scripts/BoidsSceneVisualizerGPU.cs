#pragma warning disable CS0649

using UnityEngine;

using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;

namespace Obel.Boids
{
    public struct MeshProperties
    {
        public Matrix4x4 mat;
        public Vector4 color;

        public static int Size()
        {
            return
                sizeof(float) * 4 * 4 + // matrix;
                sizeof(float) * 4;      // color;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    public struct UpdateBufferDataJob : IJobParallelFor
    {
        public float3 upAxis;
        public float3 cubeSize;

        [ReadOnly]
        public NativeArray<MeshProperties> meshProperties;

        [WriteOnly]
        public NativeArray<MeshProperties> cumputeBufferData;

        [ReadOnly]
        public NativeArray<float3> boidsPosition;

        [ReadOnly]
        public NativeArray<float3> boidsVelocity;

        [ReadOnly]
        public NativeArray<int> boidsTeamIndexes;

        [ReadOnly]
        public NativeArray<BoidTeam> teams;

        public void Execute(int boidIndex)
        {
            var prop = meshProperties[boidIndex];

            FillMatrix(ref prop.mat,
                boidsPosition[boidIndex],
                quaternion.LookRotation(boidsVelocity[boidIndex], upAxis).value,
                cubeSize * teams[boidsTeamIndexes[boidIndex]].size);

            cumputeBufferData[boidIndex] = prop;
        }

        private void FillMatrix(ref Matrix4x4 mat, float3 t, float4 r, float3 s)
        {
            mat.m00 = (1.0f  -  2.0f * (r.y * r.y + r.z * r.z)) * s.x;
            mat.m10 = (r.x * r.y + r.z * r.w) * s.x * 2.0f;
            mat.m20 = (r.x * r.z - r.y * r.w) * s.x * 2.0f;
            mat.m30 = 0.0f;

            mat.m01 = (r.x * r.y - r.z * r.w) * s.y * 2.0f;
            mat.m11 = (1.0f - 2.0f * (r.x * r.x + r.z * r.z)) * s.y;
            mat.m21 = (r.y * r.z + r.x * r.w) * s.y * 2.0f;
            mat.m31 = 0.0f;

            mat.m02 = (r.x * r.z + r.y * r.w) * s.z * 2.0f;
            mat.m12 = (r.y * r.z - r.x * r.w) * s.z * 2.0f;
            mat.m22 = (1.0f - 2.0f * (r.x * r.x + r.y * r.y)) * s.z;
            mat.m32 = 0.0f;

            mat.m03 = t.x;
            mat.m13 = t.y;
            mat.m23 = t.z;
            mat.m33 = 1.0f;
        }
    }

    public class BoidsSceneVisualizerGPU : BoidsSceneVisualizerBase
    {
        [SerializeField]
        private Mesh drawMesh;

        [SerializeField]
        private Material drawMat;

        private ComputeBuffer argsBuffer;
        private ComputeBuffer meshPropertiesBuffer;

        private Bounds meshBounds;
        private NativeArray<MeshProperties> meshPropterties;
        private NativeArray<BoidTeam> nativeTeams;
        private UpdateBufferDataJob updateBufferDataJob;

        protected override void StartVisualization()
        {
            Debug.Log(boidsCount);

            uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
            // Arguments for drawing mesh.
            // 0 == number of triangle indices, 1 == population, others are only relevant if drawing submeshes.
            args[0] = (uint)drawMesh.GetIndexCount(0);
            args[1] = (uint)boidsCount;
            args[2] = (uint)drawMesh.GetIndexStart(0);
            args[3] = (uint)drawMesh.GetBaseVertex(0);
            argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
            argsBuffer.SetData(args);

            meshBounds = new Bounds(transform.position, Vector3.one * (50 + 1));
            nativeTeams = new NativeArray<BoidTeam>(this.teams, Allocator.Persistent);
            meshPropterties = new NativeArray<MeshProperties>(boidsCount, Allocator.Persistent);

            for (int boidIndex = 0; boidIndex < boidsCount; boidIndex++)
            {
                MeshProperties props = new MeshProperties();

                switch (boidsTeamIndexes[boidIndex])
                {
                    case 0: props.color = Color.red; 
                        break;
                    case 1: props.color = Color.green;
                        break;
                    case 2: props.color = Color.blue;
                        break;
                }

                meshPropterties[boidIndex] = props;
            }

            meshPropertiesBuffer = new ComputeBuffer(boidsCount, MeshProperties.Size(), ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
            meshPropertiesBuffer.SetData(meshPropterties);

            updateBufferDataJob = new UpdateBufferDataJob
            {
                boidsPosition = this.boidsPosition,
                boidsVelocity = this.boidsVelocity,
                boidsTeamIndexes = this.boidsTeamIndexes,
                teams = this.nativeTeams,
                meshProperties = this.meshPropterties,
                upAxis = new float3(0, 1, 0),
                cubeSize = new float3(1, 1, 1)
            };
        }

        protected override void StopVisualization()
        {
            if (nativeTeams.IsCreated) nativeTeams.Dispose();

            if (meshPropterties.IsCreated) meshPropterties.Dispose();

            if (argsBuffer != null)
            {
                argsBuffer.Release();
            }

            if (meshPropertiesBuffer != null)
            {
                meshPropertiesBuffer.Release();
            }
        }

        protected override void StepVisualization()
        {
            updateBufferDataJob.cumputeBufferData = meshPropertiesBuffer.BeginWrite<MeshProperties>(0, boidsCount);

            var cumputeBufferDataJobHandler = updateBufferDataJob.Schedule(boidsCount, 64);
            cumputeBufferDataJobHandler.Complete();

            meshPropertiesBuffer.EndWrite<MeshProperties>(boidsCount);
            
            drawMat.SetBuffer("_Properties", meshPropertiesBuffer);
            Graphics.DrawMeshInstancedIndirect(drawMesh, 0, drawMat, meshBounds, argsBuffer);
        }
    }
}