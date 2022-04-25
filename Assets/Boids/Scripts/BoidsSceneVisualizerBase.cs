using System.Collections.Generic;
using UnityEngine;

using Unity.Collections;
using Unity.Mathematics;

namespace Obel.Boids
{
    public abstract class BoidsSceneVisualizerBase : MonoBehaviour
    {
        protected NativeArray<float3> boidsPosition;
        protected NativeArray<float3> boidsVelocity;
        protected NativeArray<int> boidsTeamIndexes;
        protected BoidTeam[] teams;

        protected int boidsCount;

        private void Awake()
        {
            BoidsSceneSpawner.OnSpawned += OnSpawned;
        }

        private void OnDestroy()
        {
            BoidsSceneSpawner.OnSpawned -= OnSpawned;
            StopVisualization();
        }

        private void Update()
        {
            StepVisualization();
        }

        private void OnSpawned(NativeArray<float3> positions, NativeArray<float3> velocities, NativeArray<int> teamIndexes, BoidTeam[] teams)
        {
            if (!enabled) return;

            this.boidsPosition = positions;
            this.boidsVelocity = velocities;
            this.boidsTeamIndexes = teamIndexes;
            this.teams = teams;

            boidsCount = boidsPosition.Length;

            StopVisualization();
            StartVisualization();
        }

        protected abstract void StepVisualization();

        protected abstract void StartVisualization();

        protected abstract void StopVisualization();
    }
}