using UnityEngine;

using Unity.Mathematics;
using Unity.Jobs;
using Unity.Collections;
using Unity.Burst;

namespace Obel.Boids
{
	[BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
	public struct BoidsCacheAvoidenceJob : IJobParallelFor
	{
		public float dt;
		public float avoidanceRange;
		public int cellsCount;

		[WriteOnly]
		public NativeArray<float3> boidsAvoidanceVelocities;

		[ReadOnly]
		public NativeArray<int> boidsCellIndexes;

		[ReadOnly]
		public NativeArray<int> boidsTeamIndexes;

		[ReadOnly]
		public NativeArray<float3> boidsPosition;

		[ReadOnly]
		public NativeMultiHashMap<int, int> cells;

		public void Execute(int boidIndex)
		{
			var cellIndex = boidsCellIndexes[boidIndex];
			var teamIndex = boidsTeamIndexes[boidIndex];

			var step = float3.zero;

			if (cells.TryGetFirstValue(cellIndex + (cellsCount * teamIndex), out int otherBoidIndex, out NativeMultiHashMapIterator<int> iterator))
			{
				if (otherBoidIndex != boidIndex)
					step += GetStep(boidIndex, otherBoidIndex);

				while (cells.TryGetNextValue(out otherBoidIndex, ref iterator))
				{
					if (otherBoidIndex != boidIndex)
						step += GetStep(boidIndex, otherBoidIndex);
				}
			}

			boidsAvoidanceVelocities[boidIndex] = step;
		}

        private float3 GetStep(int boidIndex, int otherBoidIndex)
        {
			var tempDelta = boidsPosition[boidIndex] - boidsPosition[otherBoidIndex];

			var tempDeltaSqr = math.lengthsq(tempDelta);

			var tempStep = float3.zero;

			if (tempDeltaSqr < avoidanceRange * avoidanceRange)
			{
				tempStep = tempDelta / math.sqrt(tempDeltaSqr);
			}

			return tempStep;
		}
	}

	[BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
	public struct BoidsMoveJob : IJobParallelFor
	{
		public BoidSimulatorParams sParams;
		public float dt;
		public int cellsCount;
		public float3 halfWorldSize;

		public NativeArray<float3> boidsPosition;

		public NativeArray<float3> boidsVelocity;

		[ReadOnly]
		public NativeArray<int> boidsCellIndexes;

		[ReadOnly]
		public NativeArray<int> boidsTeamIndexes;

		[ReadOnly]
		public NativeArray<BoidTeam> teams;

		[ReadOnly]
		public NativeArray<float3> boidsAvoidanceVelocities;

		[ReadOnly]
		public NativeArray<float3> cellsAveragePosition;

		[ReadOnly]
		public NativeArray<float3> cellsAverageVelocity;

		public void Execute(int boidIndex)
		{
			var position = boidsPosition[boidIndex];
			var velocity = boidsVelocity[boidIndex];

			var cellIndex = boidsCellIndexes[boidIndex];
			var teamIndex = boidsTeamIndexes[boidIndex];

			// Avoid world bounds
			velocity.x -= math.max(math.abs(position.x) - halfWorldSize.x + sParams.avoidanceRange, 0) * math.sign(position.x) * 5f * dt;
			velocity.y -= math.max(math.abs(position.y) - halfWorldSize.y + sParams.avoidanceRange, 0) * math.sign(position.y) * 5f * dt;
			velocity.z -= math.max(math.abs(position.z) - halfWorldSize.z + sParams.avoidanceRange, 0) * math.sign(position.z) * 5f * dt;

			// Align
			var averageVelocity = cellsAverageVelocity[cellIndex + (teamIndex * cellsCount)];
			velocity += (averageVelocity - velocity) * dt * sParams.matchVelocityRate;

			// Coherence
			var averagePosition = cellsAveragePosition[cellIndex + (teamIndex * cellsCount)];
			velocity += (averagePosition - position) * dt * sParams.coherenceRate;

			// Avoid others
			velocity += boidsAvoidanceVelocities[boidIndex] * dt * sParams.avoidanceRate;

			// Accelerations
			velocity += math.normalize(velocity) * teams[teamIndex].acceleration * dt;
			velocity *= 1.0f - 30.0f * teams[teamIndex].drag * dt;

			// Final position
			position += velocity * dt;

			boidsPosition[boidIndex] = position;
			boidsVelocity[boidIndex] = velocity;
		}
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    public struct CellsUpdateIndexesJob : IJob//ParallelFor
    {
		public float3 worldHalfSize;
		public float3 cellSize;
		public int3 cellsAxesLimits;

		public int cellsCount;
		public int teamsCount;
		public int boidsCount;

		public float dt;

		public NativeArray<int> boidsCellIndexes;

		[ReadOnly]
		public NativeArray<int> boidsTeamIndexes;

		[ReadOnly]
		public NativeArray<float3> boidsPosition;

		[ReadOnly]
		public NativeArray<float3> boidsVelocity;

		public NativeMultiHashMap<int, int> cells;

		//TODO make it job parallel
		public void Execute()
		{
			for (int i = 0; i < boidsCount; i++) Execute(i);
		}

		public void Execute(int boidIndex)
        {
			var cellIndex = boidsCellIndexes[boidIndex];
			var teamIndex = boidsTeamIndexes[boidIndex];

			var nextCellIndex = GetCellIndex(boidsPosition[boidIndex] + boidsVelocity[boidIndex] * dt);

			if (cellIndex != nextCellIndex)
			{
				cells.Remove(cellIndex + (cellsCount * teamIndex), boidIndex);
				cells.Add(nextCellIndex + (cellsCount * teamIndex), boidIndex);

				cellIndex = nextCellIndex;
			}

			boidsCellIndexes[boidIndex] = cellIndex;
		}

        private int GetCellIndex(float3 pos)
		{
			pos += worldHalfSize;

			return GetCellIndex
			(
				math.clamp((int)(pos.x / cellSize.x), 0, cellsAxesLimits.x - 1),
				math.clamp((int)(pos.y / cellSize.y), 0, cellsAxesLimits.y - 1),
				math.clamp((int)(pos.z / cellSize.z), 0, cellsAxesLimits.z - 1)
			);
		}

		private int GetCellIndex(int x, int y, int z)
		{
			return (z * cellsAxesLimits.x * cellsAxesLimits.y) + (y * cellsAxesLimits.x) + x;
		}
	}

	[BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
	public struct CellsCacheVelocityJob : IJobParallelFor
    {
		[ReadOnly]
		public NativeArray<float3> boidsPosition;

		[ReadOnly]
		public NativeArray<float3> boidsVelocity;

		[ReadOnly]
		public NativeMultiHashMap<int, int> cells;

		[WriteOnly]
		public NativeArray<float3> cellsAveragePosition;
		[WriteOnly]
		public NativeArray<float3> cellsAverageVelocity;

		public int cellsCount;
        public void Execute(int cellIndex)
        {
			var cellAveragePosition = float3.zero;
			var cellAverageVelocity = float3.zero;

			var cellBoidsCount = 0;

			if (cells.TryGetFirstValue(cellIndex, out int boidIndex, out NativeMultiHashMapIterator<int> iterator))
			{
				cellAveragePosition += boidsPosition[boidIndex];
				cellAverageVelocity += boidsVelocity[boidIndex];
				cellBoidsCount++;
				while (cells.TryGetNextValue(out boidIndex, ref iterator))
				{
					cellAveragePosition += boidsPosition[boidIndex];
					cellAverageVelocity += boidsVelocity[boidIndex];
					cellBoidsCount++;
				}
			}
			
			cellsAverageVelocity[cellIndex] = cellAverageVelocity / cellBoidsCount;
			cellsAveragePosition[cellIndex] = cellAveragePosition * 1.0f / cellBoidsCount;
		}
    }

    public struct BoidSimulatorParams
	{
		public float initialVelocity;
		public float matchVelocityRate;
		public float avoidanceRange;
		public float avoidanceRate;
		public float coherenceRate;
		public float viewRange;
	}

	public class BoidParallelSimulator
	{
		private BoidSimulatorParams sParams = new BoidSimulatorParams()
		{
			initialVelocity = 2.0f,
			matchVelocityRate = 4.0f,

			avoidanceRange = 2.0f,
			avoidanceRate = 5.0f,

			coherenceRate = 2.0f,

			viewRange = 3.0f
		};

		public NativeArray<float3> Positions => boidsPosition;
		public NativeArray<float3> Velocities => boidsVelocity;
		public NativeArray<int> TeamIndexes => boidsTeamIndexes;

		private float WORLD_CELL_DIVISION = 0.6f;
		private float WORLD_TIME_SCALE = 1.0f;

		private float3 worldSize;
		private float3 worldHalfSize;

		private int teamsCount;
		private int boidsCount;
		private int cellsCount;

		private NativeArray<float3> boidsPosition;
		private NativeArray<float3> boidsVelocity;
		private NativeArray<int> boidsTeamIndexes;
		private NativeArray<int> boidsCellIndexes;

		private NativeArray<BoidTeam> teams;

		private CellsUpdateIndexesJob cellsUpdateIndexesJob;
		private CellsCacheVelocityJob cellsCacheVelocityJob;
		private BoidsCacheAvoidenceJob boidsCacheAvoidanceJob;
		private BoidsMoveJob boidsMoveJob;

		private NativeArray<float3> boidsAvoidanceVelocity;
		private NativeArray<float3> cellsAveragePosition;
		private NativeArray<float3> cellsAverageVelocity;
		private NativeMultiHashMap<int, int> cells;

		private NativeMultiHashMap<int, int> cellsIndexesToRemove;

		private int3 cellsAxesLimits;
		private float3 cellSize;

		public void StartSimulation(float3 worldSize, int boidsCount, BoidTeam[] teams)
		{
			StopSimulation();

			this.teams = new NativeArray<BoidTeam>(teams, Allocator.Persistent);
			this.teamsCount = teams.Length;

			this.worldSize = worldSize;
			this.worldHalfSize = worldSize * 0.5f;

			this.boidsCount = boidsCount;

			InitBoids();
			InitCells();
			InitJobs();
		}

		private void InitBoids()
        {
			boidsPosition = new NativeArray<float3>(boidsCount, Allocator.Persistent);
			boidsVelocity = new NativeArray<float3>(boidsCount, Allocator.Persistent);
			boidsTeamIndexes = new NativeArray<int>(boidsCount, Allocator.Persistent);
			boidsCellIndexes = new NativeArray<int>(boidsCount, Allocator.Persistent);

			boidsAvoidanceVelocity = new NativeArray<float3>(boidsCount, Allocator.Persistent);

			var halfSpawnRange = worldSize * .5f - new float3(3, 3, 3);

			for (int boidIndex = 0; boidIndex < boidsCount; boidIndex++)
			{
				boidsPosition[boidIndex] = new float3
				(
					UnityEngine.Random.Range(-halfSpawnRange.x, halfSpawnRange.x),
					UnityEngine.Random.Range(-halfSpawnRange.y, halfSpawnRange.y),
					UnityEngine.Random.Range(-halfSpawnRange.z, halfSpawnRange.z)
				);

				boidsVelocity[boidIndex] = UnityEngine.Random.onUnitSphere * sParams.initialVelocity;

				boidsTeamIndexes[boidIndex] = UnityEngine.Random.Range(0, teams.Length);
			}
		}

		private void InitCells()
		{
			cellsAxesLimits = new int3(worldSize / 3.0f * WORLD_CELL_DIVISION);
			cellSize = new float3(worldSize / cellsAxesLimits);
			cellsCount = cellsAxesLimits.x * cellsAxesLimits.y * cellsAxesLimits.z;

			cells = new NativeMultiHashMap<int, int>(cellsCount * teamsCount, Allocator.Persistent);
			cellsAveragePosition = new NativeArray<float3>(cellsCount * teamsCount, Allocator.Persistent);
			cellsAverageVelocity = new NativeArray<float3>(cellsCount * teamsCount, Allocator.Persistent);
			cellsIndexesToRemove = new NativeMultiHashMap<int, int>(cellsCount * teamsCount, Allocator.Persistent);
		}

		private void InitJobs()
		{
			boidsCacheAvoidanceJob = new BoidsCacheAvoidenceJob
			{
				cellsCount = this.cellsCount,

				boidsPosition = this.boidsPosition,
				boidsCellIndexes = this.boidsCellIndexes,
				boidsTeamIndexes = this.boidsTeamIndexes,

				cells = this.cells,
				boidsAvoidanceVelocities = this.boidsAvoidanceVelocity
			};

			boidsMoveJob = new BoidsMoveJob
			{
				cellsCount = this.cellsCount,
				halfWorldSize = this.worldHalfSize,

				boidsPosition = this.boidsPosition,
				boidsVelocity = this.boidsVelocity,
				boidsCellIndexes = this.boidsCellIndexes,
				boidsTeamIndexes = this.boidsTeamIndexes,

				teams = this.teams,
				cellsAveragePosition = this.cellsAveragePosition,
				cellsAverageVelocity = this.cellsAverageVelocity,
				boidsAvoidanceVelocities = this.boidsAvoidanceVelocity
			};

			cellsUpdateIndexesJob = new CellsUpdateIndexesJob
			{
				cellsCount = this.cellsCount,
				teamsCount = this.teamsCount,
				boidsCount = this.boidsCount,
				worldHalfSize = this.worldHalfSize,
				cellSize = this.cellSize,

				boidsPosition = this.boidsPosition,
				boidsVelocity = this.boidsVelocity,
				boidsCellIndexes = this.boidsCellIndexes,
				boidsTeamIndexes = this.boidsTeamIndexes,

				cells = this.cells,
				cellsAxesLimits = this.cellsAxesLimits
			};

			cellsCacheVelocityJob = new CellsCacheVelocityJob
			{
				cellsCount = this.cellsCount,

				boidsPosition = this.boidsPosition,
				boidsVelocity = this.boidsVelocity,

				cells = this.cells,
				cellsAveragePosition = this.cellsAveragePosition,
				cellsAverageVelocity = this.cellsAverageVelocity,
			};
		}

		public void StepSimulation(float dt)
		{
			StepSimulationInternal(dt);
		}

		private void StepSimulationInternal(float dt)
		{
			dt *= WORLD_TIME_SCALE;

			cellsUpdateIndexesJob.dt = dt;

			boidsCacheAvoidanceJob.dt = dt;
			boidsCacheAvoidanceJob.avoidanceRange = sParams.avoidanceRange;

			boidsMoveJob.dt = dt;
			boidsMoveJob.sParams = this.sParams;

			var cellsUpdateIndexesJobHandler  = cellsUpdateIndexesJob.Schedule();
			var cellsCacheVelocityJobHandler  = cellsCacheVelocityJob.Schedule(cellsCount * teamsCount, cellsCount / 8, cellsUpdateIndexesJobHandler);
			var boidsCacheAvoidanceJobHandler = boidsCacheAvoidanceJob.Schedule(boidsCount, 64, cellsCacheVelocityJobHandler);
			var boidsMoveJobHandler = boidsMoveJob.Schedule(boidsCount, 64, boidsCacheAvoidanceJobHandler);

			boidsMoveJobHandler.Complete();
		}

		public void StopSimulation()
		{
			if (boidsPosition.IsCreated) boidsPosition.Dispose();
			if (boidsVelocity.IsCreated) boidsVelocity.Dispose();
			if (boidsTeamIndexes.IsCreated) boidsTeamIndexes.Dispose();
			if (boidsCellIndexes.IsCreated) boidsCellIndexes.Dispose();

			if (teams.IsCreated) teams.Dispose();
			if (cells.IsCreated) cells.Dispose();

			if (cellsAveragePosition.IsCreated) cellsAveragePosition.Dispose();
			if (cellsAverageVelocity.IsCreated) cellsAverageVelocity.Dispose();
			if (cellsIndexesToRemove.IsCreated) cellsIndexesToRemove.Dispose();

			if (boidsAvoidanceVelocity.IsCreated) boidsAvoidanceVelocity.Dispose();
		}

		public void DrawGUI()
		{
			var rect = new Rect(20, 20, 300, 360);

			GUI.Box(rect, $"{boidsCount} BOIDS");

			GUILayout.BeginArea(rect);

			GUILayout.BeginHorizontal();
			GUILayout.Label("  ");
			GUILayout.BeginHorizontal();

			GUILayout.BeginHorizontal();
			GUILayout.BeginVertical();

			GUILayout.Space(32);

			GUILayout.Label($"VELOCITY RATE {sParams.matchVelocityRate.ToString("0.0")}");
			sParams.matchVelocityRate = GUILayout.HorizontalSlider(sParams.matchVelocityRate, 0.1f, 10);
			GUILayout.Space(4);

			GUILayout.Label($"AVOIDANCE RANGE {sParams.avoidanceRange.ToString("0.0")}");
			sParams.avoidanceRange = GUILayout.HorizontalSlider(sParams.avoidanceRange, 0.1f, 10);
			GUILayout.Space(4);

			GUILayout.Label($"AVOIDANCE RATE {sParams.avoidanceRate.ToString("0.0")}");
			sParams.avoidanceRate = GUILayout.HorizontalSlider(sParams.avoidanceRate, 0.1f, 10);
			GUILayout.Space(4);

			GUILayout.Label($"COHERENCE RATE {sParams.coherenceRate.ToString("0.0")}");
			sParams.coherenceRate = GUILayout.HorizontalSlider(sParams.coherenceRate, 0.1f, 10);
			GUILayout.Space(4);

			GUILayout.Label($"VIEW RANGE {sParams.viewRange.ToString("0.0")}");
			sParams.viewRange = GUILayout.HorizontalSlider(sParams.viewRange, 0.5f, 10);
			GUILayout.Space(4);

			GUILayout.Label($"CELL DIVISION {WORLD_CELL_DIVISION.ToString("0.0")}               (restart required)");
			WORLD_CELL_DIVISION = GUILayout.HorizontalSlider(WORLD_CELL_DIVISION, 0.1f, 1.5f);
			GUILayout.Space(4);

			GUILayout.Label($"TIME SCALE {WORLD_TIME_SCALE.ToString("0.0")}");
			WORLD_TIME_SCALE = GUILayout.HorizontalSlider(WORLD_TIME_SCALE, 0, 10);

			GUILayout.EndVertical();
			GUILayout.EndHorizontal();

			GUILayout.BeginHorizontal();
			GUILayout.Label("  ");
			GUILayout.BeginHorizontal();

			GUILayout.EndArea();
		}

#if UNITY_EDITOR
		public void DrawGizmos()
		{
			var density = (float)boidsCount / cellsCount;

			for (int cellIndex = 0; cellIndex < cellsCount; cellIndex++)
			{
				var cellPosition = GetCellPosition(cellIndex);

				for (int teamIndex = 0; teamIndex < teamsCount; teamIndex++)
				{
					var cellColor = Color.white;

					switch (teamIndex)
					{
						case 0:
							cellColor = Color.red;
							break;
						case 1:
							cellColor = Color.green;
							break;
						case 2:
							cellColor = Color.blue;
							break;
					}

					var averagePos = cellsAveragePosition[cellIndex + (teamIndex * cellsCount)];
					Gizmos.color = cellColor;
					Gizmos.DrawSphere(averagePos, 0.2f);
					Gizmos.DrawLine(averagePos, averagePos + cellsAverageVelocity[cellIndex + (teamIndex * cellsCount)]);

					cellColor.a = cells.CountValuesForKey(cellIndex + (cellsCount * teamIndex)) / density / WORLD_CELL_DIVISION * 0.2f;
					Gizmos.color = cellColor;
					Gizmos.DrawWireCube(cellPosition, cellSize);
				}
			}
		}

		private float3 GetCellPosition(int index)
		{
			float3 pos = GetCellLocalPosition(index);

			pos.x *= cellSize.x;
			pos.y *= cellSize.y;
			pos.z *= cellSize.z;

			pos -= worldHalfSize - cellSize * 0.5f;

			return pos;
		}

		private int3 GetCellLocalPosition(int index)
		{
			var axis = GetCell3DIndexes(index);
			return new int3(axis[0], axis[1], axis[2]);
		}

		private int[] GetCell3DIndexes(int index)
		{
			int z = index / (cellsAxesLimits.x * cellsAxesLimits.y);
			index -= (z * cellsAxesLimits.x * cellsAxesLimits.y);
			int y = index / cellsAxesLimits.x;
			int x = index % cellsAxesLimits.x;
			return new int[] { x, y, z };
		}
#endif
	}
}