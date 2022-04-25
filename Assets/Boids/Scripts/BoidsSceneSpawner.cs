using UnityEngine;

using Unity.Collections;
using Unity.Mathematics;

namespace Obel.Boids
{
	public class BoidsSceneSpawner : MonoBehaviour
	{
		private const float BOID_DENSITY = 4f;
		private const int ROUND_WORLD_SIZE_TO_MULTIPLES_OF = 5;

		public delegate void OnSpawnedDelegate
		(
			NativeArray<float3> positions, 
			NativeArray<float3> velocities, 
			NativeArray<int> teamIndexes, 
			BoidTeam[] teams
		);

		public static OnSpawnedDelegate OnSpawned;

		private static readonly int[] boidCounts = new int[]
		{
			64,
			256,
			1024,
			4096,
			8192,
			16384,
			32768,
			65536,
			262144
		};


		private static readonly BoidTeam[] teams = new BoidTeam[]
		{
			// Red
			new BoidTeam
			{
				acceleration = 4,
				drag = .02f,
				size = 1
			},
			
			// Green
			new BoidTeam
			{
				acceleration = 4,
				drag = .03f,
				size = 0.5f
			},

			// Blue
			new BoidTeam
			{
				acceleration = 11,
				drag = .04f,
				size = 0.33f
			},
		};

		private readonly BoidParallelSimulator boidSim = new BoidParallelSimulator();
		private void Start()
		{
			ResetSetup(boidCounts[0]);
		}

		private void Update()
		{
			// Reset setup when number keys are pressed
			for (int i = 0; i < boidCounts.Length; i++)
			{
				if (Input.GetKeyDown(KeyCode.Alpha1 + i))
				{
					ResetSetup(boidCounts[i]);
					break;
				}
			}

			boidSim.StepSimulation(Time.deltaTime);
		}

		void OnDestroy()
		{
			boidSim.StopSimulation();
		}

		private void ResetSetup(int boidsCount)
		{
			// Decide world size based on boid count and density
			int worldSize = Mathf.CeilToInt(Mathf.Pow(boidsCount, 1f / 3) * BOID_DENSITY / ROUND_WORLD_SIZE_TO_MULTIPLES_OF) * ROUND_WORLD_SIZE_TO_MULTIPLES_OF;

			// Reset boid simulator
			boidSim.StartSimulation(new Vector3(worldSize, worldSize, worldSize), boidsCount, teams);

			OnSpawned?.Invoke(boidSim.Positions, boidSim.Velocities, boidSim.TeamIndexes, teams);
		}

		private void OnGUI()
		{
			boidSim.DrawGUI();
		}

#if UNITY_EDITOR
		private void OnDrawGizmos()
		{
			boidSim.DrawGizmos();
		}
#endif
	}
}