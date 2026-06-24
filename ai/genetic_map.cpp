#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <stdexcept>

using namespace std;

const int WIDTH = 30;
const int HEIGHT = 30;

const int FLOOR = 0;
const int WALL = 1;

struct Position {
    int row;
    int col;
};

struct ScoredMap {
    double score;
    vector<vector<int>> grid;
};

mt19937 rng;


void seedRng(uint64_t seed) {
    seed_seq sequence{
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32)
    };
    rng.seed(sequence);
}


uint64_t makeDynamicSeed() {
    random_device device;
    const uint64_t clockValue = static_cast<uint64_t>(
        chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    return clockValue
        ^ (static_cast<uint64_t>(device()) << 32)
        ^ static_cast<uint64_t>(device());
}


int randomInt(int low, int high) {
    uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}


double randomDouble() {
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}


vector<vector<int>> createRandomMap(double wallRate = 0.28) {
    vector<vector<int>> grid(HEIGHT, vector<int>(WIDTH, FLOOR));

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (randomDouble() < wallRate) {
                grid[i][j] = WALL;
            } else {
                grid[i][j] = FLOOR;
            }
        }
    }

    return grid;
}


vector<Position> getNeighbors(Position pos, const vector<vector<int>>& grid) {
    int height = grid.size();
    int width = grid[0].size();

    vector<Position> result;

    vector<pair<int, int>> moves = {
        {-1, 0},
        {1, 0},
        {0, -1},
        {0, 1}
    };

    for (auto move : moves) {
        int nextRow = (pos.row + move.first + height) % height;
        int nextCol = (pos.col + move.second + width) % width;

        if (grid[nextRow][nextCol] == FLOOR) {
            result.push_back({nextRow, nextCol});
        }
    }

    return result;
}


vector<Position> getFloorCells(const vector<vector<int>>& grid) {
    vector<Position> cells;

    for (int i = 0; i < (int)grid.size(); i++) {
        for (int j = 0; j < (int)grid[0].size(); j++) {
            if (grid[i][j] == FLOOR) {
                cells.push_back({i, j});
            }
        }
    }

    return cells;
}


int bfsCount(Position start, const vector<vector<int>>& grid) {
    int height = grid.size();
    int width = grid[0].size();

    vector<vector<bool>> visited(height, vector<bool>(width, false));
    queue<Position> q;

    q.push(start);
    visited[start.row][start.col] = true;

    int count = 0;

    while (!q.empty()) {
        Position cur = q.front();
        q.pop();

        count++;

        vector<Position> neighbors = getNeighbors(cur, grid);

        for (Position nxt : neighbors) {
            if (!visited[nxt.row][nxt.col]) {
                visited[nxt.row][nxt.col] = true;
                q.push(nxt);
            }
        }
    }

    return count;
}


int bfsDistance(Position start, Position end, const vector<vector<int>>& grid) {
    if (start.row == end.row && start.col == end.col) {
        return 0;
    }

    int height = grid.size();
    int width = grid[0].size();

    vector<vector<bool>> visited(height, vector<bool>(width, false));
    queue<pair<Position, int>> q;

    q.push({start, 0});
    visited[start.row][start.col] = true;

    while (!q.empty()) {
        Position cur = q.front().first;
        int dist = q.front().second;
        q.pop();

        vector<Position> neighbors = getNeighbors(cur, grid);

        for (Position nxt : neighbors) {
            if (nxt.row == end.row && nxt.col == end.col) {
                return dist + 1;
            }

            if (!visited[nxt.row][nxt.col]) {
                visited[nxt.row][nxt.col] = true;
                q.push({nxt, dist + 1});
            }
        }
    }

    return -1;
}


double fitness(const vector<vector<int>>& grid) {
    vector<Position> floorCells = getFloorCells(grid);

    if (floorCells.size() < 2) {
        return 0;
    }

    int height = grid.size();
    int width = grid[0].size();
    int totalCells = height * width;

    int wallCount = totalCells - floorCells.size();
    double wallRatio = (double)wallCount / totalCells;

    // 1. connectivity score
    int reachable = bfsCount(floorCells[0], grid);
    double reachableRatio = (double)reachable / floorCells.size();
    double reachScore = reachableRatio * 40.0;

    // 2. wall ratio score
    double idealWallRatio = 0.28;
    double wallScore = max(
        0.0,
        1.0 - abs(wallRatio - idealWallRatio) / idealWallRatio
    ) * 20.0;

    // 3. junction score and dead-end penalty
    int junctions = 0;
    int deadEnds = 0;

    for (Position cell : floorCells) {
        int neighborCount = getNeighbors(cell, grid).size();

        if (neighborCount >= 3) {
            junctions++;
        }

        if (neighborCount <= 1) {
            deadEnds++;
        }
    }

    double junctionRatio = (double)junctions / floorCells.size();
    double junctionScore = min(junctionRatio * 100.0, 20.0);

    double deadEndRatio = (double)deadEnds / floorCells.size();
    double deadEndPenalty = deadEndRatio * 20.0;

    // 4. path distance score
    vector<int> distances;

    for (int i = 0; i < 30; i++) {
        Position a = floorCells[randomInt(0, floorCells.size() - 1)];
        Position b = floorCells[randomInt(0, floorCells.size() - 1)];

        int d = bfsDistance(a, b, grid);

        if (d != -1) {
            distances.push_back(d);
        }
    }

    double pathScore = 0.0;

    if (!distances.empty()) {
        double sum = 0.0;

        for (int d : distances) {
            sum += d;
        }

        double avgDist = sum / distances.size();
        pathScore = max(0.0, 1.0 - abs(avgDist - 20.0) / 20.0) * 20.0;
    }

    double totalScore =
        reachScore
        + wallScore
        + junctionScore
        + pathScore
        - deadEndPenalty;

    return totalScore;
}


vector<vector<vector<int>>> createPopulation(int size) {
    vector<vector<vector<int>>> population;

    for (int i = 0; i < size; i++) {
        population.push_back(createRandomMap());
    }

    return population;
}


vector<vector<int>> selectParent(const vector<ScoredMap>& scoredMaps) {
    vector<ScoredMap> candidates;

    for (int i = 0; i < 3; i++) {
        int index = randomInt(0, scoredMaps.size() - 1);
        candidates.push_back(scoredMaps[index]);
    }

    sort(candidates.begin(), candidates.end(), [](const ScoredMap& a, const ScoredMap& b) {
        return a.score > b.score;
    });

    return candidates[0].grid;
}


vector<vector<int>> crossover(
    const vector<vector<int>>& parentA,
    const vector<vector<int>>& parentB
) {
    vector<vector<int>> child;

    if (randomDouble() < 0.5) {
        int cut = randomInt(1, HEIGHT - 2);

        for (int i = 0; i < cut; i++) {
            child.push_back(parentA[i]);
        }

        for (int i = cut; i < HEIGHT; i++) {
            child.push_back(parentB[i]);
        }
    } else {
        int cut = randomInt(1, WIDTH - 2);

        for (int i = 0; i < HEIGHT; i++) {
            vector<int> row;

            for (int j = 0; j < cut; j++) {
                row.push_back(parentA[i][j]);
            }

            for (int j = cut; j < WIDTH; j++) {
                row.push_back(parentB[i][j]);
            }

            child.push_back(row);
        }
    }

    return child;
}


vector<vector<int>> mutate(const vector<vector<int>>& grid, double mutationRate = 0.01) {
    vector<vector<int>> newGrid = grid;

    for (int i = 0; i < (int)newGrid.size(); i++) {
        for (int j = 0; j < (int)newGrid[0].size(); j++) {
            if (randomDouble() < mutationRate) {
                newGrid[i][j] = 1 - newGrid[i][j];
            }
        }
    }

    return newGrid;
}


vector<vector<int>> generateMapGA(
    int popSize = 30,
    int generations = 20,
    double mutationRate = 0.01,
    int eliteNum = 5
) {
    vector<vector<vector<int>>> population = createPopulation(popSize);

    vector<vector<int>> bestMap;
    double bestScore = -1.0;

    for (int gen = 0; gen < generations; gen++) {
        vector<ScoredMap> scored;

        for (auto grid : population) {
            double score = fitness(grid);

            scored.push_back({score, grid});

            if (score > bestScore) {
                bestScore = score;
                bestMap = grid;
            }
        }

        sort(scored.begin(), scored.end(), [](const ScoredMap& a, const ScoredMap& b) {
            return a.score > b.score;
        });

        cout << "Generation " << gen + 1
             << " Current Best: " << scored[0].score
             << " Global Best: " << bestScore
             << endl;

        vector<vector<vector<int>>> nextPopulation;

        for (int i = 0; i < eliteNum; i++) {
            nextPopulation.push_back(scored[i].grid);
        }

        while ((int)nextPopulation.size() < popSize) {
            vector<vector<int>> p1 = selectParent(scored);
            vector<vector<int>> p2 = selectParent(scored);

            vector<vector<int>> child = crossover(p1, p2);
            child = mutate(child, mutationRate);

            nextPopulation.push_back(child);
        }

        population = nextPopulation;
    }

    return bestMap;
}


pair<Position, Position> getValidSpawnPoints(
    const vector<vector<int>>& grid,
    int minDist = 30,
    int maxDist = 40,
    int maxAttempts = 5000
) {
    vector<Position> floorCells = getFloorCells(grid);

    if (floorCells.size() < 2) {
        cout << "Not enough floor cells to generate spawn points." << endl;
        exit(1);
    }

    bool hasBackup = false;
    Position backupHuman;
    Position backupMonster;
    int bestDistance = -1;

    for (int attempt = 0; attempt < maxAttempts; attempt++) {
        Position human = floorCells[randomInt(0, floorCells.size() - 1)];
        Position monster = floorCells[randomInt(0, floorCells.size() - 1)];

        if (human.row == monster.row && human.col == monster.col) {
            continue;
        }

        if ((int)getNeighbors(human, grid).size() < 2) {
            continue;
        }

        if ((int)getNeighbors(monster, grid).size() < 2) {
            continue;
        }

        int distance = bfsDistance(human, monster, grid);

        if (distance == -1) {
            continue;
        }

        if (distance >= minDist && distance <= maxDist) {
            return {human, monster};
        }

        if (distance > bestDistance) {
            bestDistance = distance;
            backupHuman = human;
            backupMonster = monster;
            hasBackup = true;
        }
    }

    if (hasBackup) {
        return {backupHuman, backupMonster};
    }

    cout << "Failed to find valid spawn points." << endl;
    exit(1);
}


void saveMapToTxt(
    const vector<vector<int>>& grid,
    const string& filePath,
    Position human,
    Position monster
) {
    ofstream fout(filePath);

    if (!fout) {
        cout << "Failed to save map file. Please make sure the map folder exists." << endl;
        exit(1);
    }

    fout << grid.size() << " " << grid[0].size() << endl;

    for (int i = 0; i < (int)grid.size(); i++) {
        for (int j = 0; j < (int)grid[0].size(); j++) {
            fout << grid[i][j];

            if (j < (int)grid[0].size() - 1) {
                fout << " ";
            }
        }
        fout << endl;
    }

    fout << human.row << " " << human.col << endl;
    fout << monster.row << " " << monster.col << endl;

    fout.close();

    cout << "TXT map saved to: " << filePath << endl;
}

void printMap(
    const vector<vector<int>>& grid,
    Position human,
    Position monster
) {
    for (int i = 0; i < (int)grid.size(); i++) {
        for (int j = 0; j < (int)grid[0].size(); j++) {
            if (human.row == i && human.col == j) {
                cout << "H";
            } else if (monster.row == i && monster.col == j) {
                cout << "M";
            } else if (grid[i][j] == WALL) {
                cout << "#";
            } else {
                cout << ".";
            }
        }
        cout << endl;
    }
}


int main(int argc, char* argv[]) {
    uint64_t seed = makeDynamicSeed();
    try {
        if (argc == 2) {
            seed = stoull(argv[1]);
        } else if (argc == 3 && string(argv[1]) == "--seed") {
            seed = stoull(argv[2]);
        } else if (argc != 1) {
            cerr << "Usage: genetic_map [--seed SEED]" << endl;
            return 2;
        }
    } catch (const exception& exc) {
        cerr << "Invalid seed: " << exc.what() << endl;
        return 2;
    }
    seedRng(seed);
    cout << "Seed: " << seed << endl;

    vector<vector<int>> generatedMap = generateMapGA(
        30,
        20,
        0.015,
        3
    );

    auto spawnPoints = getValidSpawnPoints(generatedMap);
    Position human = spawnPoints.first;
    Position monster = spawnPoints.second;

    printMap(generatedMap, human, monster);

    cout << "Fitness: " << fitness(generatedMap) << endl;
    cout << "Human spawn: " << human.row << " " << human.col << endl;
    cout << "Monster spawn: " << monster.row << " " << monster.col << endl;
    cout << "BFS spawn distance: " << bfsDistance(human, monster, generatedMap) << endl;

    saveMapToTxt(
        generatedMap,
        "map/generated_map.txt",
        human,
        monster
    );

    return 0;
}
