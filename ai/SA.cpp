#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

constexpr int MAP_SIZE = 30;
constexpr int MONSTER_MOVE_STEPS = 2;
constexpr int EARLY_WALL_STEPS = 6;
constexpr int MAX_PATH_LEN = 140;
constexpr double BACKTRACK_PENALTY = 10.0;

struct Point {
    int x;
    int y;
};

struct Config {
    int innerLoop = 20;
    double initialTemperature = 80.0;
    double minimumTemperature = 0.5;
    double coolingRate = 0.94;
    double moveDistanceWeight = 3.0;
    double wallPenalty = 100.0;
    double earlyWallPenalty = 10000.0;
    int greedySteps = 80;
    int segmentSpan = 30;
    double earlyBias = 0.70;
    string mapFile = "../map/generated_map.txt";
    bool seeded = false;
    uint32_t seed = 0;
};

Config config;
int humanX;
int humanY;
int monsterX;
int monsterY;
bool walls[MAP_SIZE][MAP_SIZE];
bool hasPreviousMove = false;
Point previousMoveFrom{};
Point previousMoveTo{};
mt19937 rng;

int wrap(int value) {
    return (value + MAP_SIZE) % MAP_SIZE;
}

bool samePoint(const Point& a, const Point& b) {
    return a.x == b.x && a.y == b.y;
}

bool available(const Point& p) {
    return !walls[p.x][p.y];
}

vector<Point> nextPoints(const Point& p) {
    return {
        {wrap(p.x + 1), p.y},
        {wrap(p.x - 1), p.y},
        {p.x, wrap(p.y + 1)},
        {p.x, wrap(p.y - 1)},
    };
}

bool adjacent(const Point& a, const Point& b) {
    for (const Point& next : nextPoints(a)) {
        if (samePoint(next, b)) {
            return true;
        }
    }
    return false;
}

int torusDistance(const Point& a, const Point& b) {
    int dx = abs(a.x - b.x);
    int dy = abs(a.y - b.y);
    return min(dx, MAP_SIZE - dx) + min(dy, MAP_SIZE - dy);
}

int signedShortestDelta(int from, int to) {
    int forward = wrap(to - from);
    int backward = forward - MAP_SIZE;
    if (abs(forward) < abs(backward)) {
        return forward;
    }
    if (abs(backward) < abs(forward)) {
        return backward;
    }
    uniform_int_distribution<int> pick(0, 1);
    return pick(rng) == 0 ? forward : backward;
}

vector<Point> randomShortestPath(const Point& start, const Point& target) {
    vector<pair<int, int>> moves;
    int dx = signedShortestDelta(start.x, target.x);
    int dy = signedShortestDelta(start.y, target.y);

    for (int i = 0; i < abs(dx); ++i) {
        moves.push_back({dx > 0 ? 1 : -1, 0});
    }
    for (int i = 0; i < abs(dy); ++i) {
        moves.push_back({0, dy > 0 ? 1 : -1});
    }
    shuffle(moves.begin(), moves.end(), rng);

    vector<Point> path{start};
    Point current = start;
    for (const auto& [stepX, stepY] : moves) {
        current = {wrap(current.x + stepX), wrap(current.y + stepY)};
        path.push_back(current);
    }
    return path;
}

Point actualMovePoint(const vector<Point>& path) {
    int step = min(MONSTER_MOVE_STEPS, static_cast<int>(path.size()) - 1);
    return path[step];
}

bool firstMovesAreLegal(const vector<Point>& path) {
    int last = min(MONSTER_MOVE_STEPS, static_cast<int>(path.size()) - 1);
    for (int i = 1; i <= last; ++i) {
        if (!available(path[i]) || !adjacent(path[i - 1], path[i])) {
            return false;
        }
    }
    return true;
}

bool validCandidate(const vector<Point>& path, const Point& start, const Point& target) {
    if (path.empty() || static_cast<int>(path.size()) > MAX_PATH_LEN) {
        return false;
    }
    if (!samePoint(path.front(), start) || !samePoint(path.back(), target)) {
        return false;
    }
    for (int i = 1; i < static_cast<int>(path.size()); ++i) {
        if (!adjacent(path[i - 1], path[i])) {
            return false;
        }
    }
    return true;
}

vector<Point> chooseGreedyCandidates(
    const Point& current,
    const Point& previous,
    const Point& target,
    const bool visited[MAP_SIZE][MAP_SIZE]
) {
    vector<Point> candidates = nextPoints(current);

    auto collect = [&](bool requireUnvisited, bool forbidBacktrack) {
        vector<Point> result;
        for (const Point& candidate : candidates) {
            if (!available(candidate)) {
                continue;
            }
            if (requireUnvisited && visited[candidate.x][candidate.y]) {
                continue;
            }
            if (forbidBacktrack && samePoint(candidate, previous)) {
                continue;
            }
            result.push_back(candidate);
        }
        return result;
    };

    vector<Point> filtered = collect(true, true);
    if (filtered.empty()) {
        filtered = collect(false, true);
    }
    if (filtered.empty()) {
        filtered = collect(false, false);
    }
    if (filtered.empty()) {
        return filtered;
    }

    int bestDistance = numeric_limits<int>::max();
    for (const Point& candidate : filtered) {
        bestDistance = min(bestDistance, torusDistance(candidate, target));
    }

    vector<Point> nearest;
    for (const Point& candidate : filtered) {
        if (torusDistance(candidate, target) == bestDistance) {
            nearest.push_back(candidate);
        }
    }
    shuffle(nearest.begin(), nearest.end(), rng);
    return nearest;
}

vector<Point> rebuildSegment(const Point& left, const Point& right) {
    vector<Point> segment{left};
    bool visited[MAP_SIZE][MAP_SIZE] = {};
    visited[left.x][left.y] = true;

    Point previous = left;
    Point current = left;
    for (int step = 0; step < config.greedySteps && !samePoint(current, right); ++step) {
        vector<Point> candidates =
            chooseGreedyCandidates(current, previous, right, visited);
        if (candidates.empty()) {
            break;
        }
        Point next = candidates.front();
        previous = current;
        current = next;
        segment.push_back(current);
        visited[current.x][current.y] = true;
    }

    if (!samePoint(current, right)) {
        vector<Point> closing = randomShortestPath(current, right);
        segment.insert(segment.end(), closing.begin() + 1, closing.end());
    }
    return segment;
}

vector<Point> mutatePath(const vector<Point>& path) {
    if (path.size() < 2) {
        return path;
    }

    int maxLeft = static_cast<int>(path.size()) - 2;
    uniform_real_distribution<double> realPick(0.0, 1.0);
    bool earlyMutation = realPick(rng) < config.earlyBias;
    int left = 0;
    if (!earlyMutation) {
        uniform_int_distribution<int> leftPick(0, maxLeft);
        left = leftPick(rng);
    }

    int right = static_cast<int>(path.size()) - 1;
    if (!earlyMutation || realPick(rng) >= 0.5) {
        int maxRight = min(
            static_cast<int>(path.size()) - 1,
            left + max(1, config.segmentSpan)
        );
        uniform_int_distribution<int> rightPick(left + 1, maxRight);
        right = rightPick(rng);
    }

    vector<Point> replacement = rebuildSegment(path[left], path[right]);
    vector<Point> candidate;
    candidate.reserve(path.size() + replacement.size());
    candidate.insert(candidate.end(), path.begin(), path.begin() + left + 1);
    candidate.insert(candidate.end(), replacement.begin() + 1, replacement.end());
    candidate.insert(candidate.end(), path.begin() + right + 1, path.end());

    Point start{monsterX, monsterY};
    Point target{humanX, humanY};
    return validCandidate(candidate, start, target) ? candidate : path;
}

double wallCost(const vector<Point>& path) {
    double cost = 0.0;
    for (int i = 1; i < static_cast<int>(path.size()); ++i) {
        if (!available(path[i])) {
            cost += i <= EARLY_WALL_STEPS
                ? config.earlyWallPenalty
                : config.wallPenalty;
        }
    }
    return cost;
}

double scorePath(const vector<Point>& path) {
    Point target{humanX, humanY};
    Point moved = actualMovePoint(path);
    double score = static_cast<double>(path.size());
    score += config.moveDistanceWeight * torusDistance(moved, target);
    score += wallCost(path);
    if (hasPreviousMove && samePoint(moved, previousMoveFrom)) {
        score += BACKTRACK_PENALTY;
    }
    return score;
}

vector<Point> simulatedAnnealing(const vector<Point>& initialPath) {
    vector<Point> current = initialPath;
    vector<Point> best = current;
    vector<Point> bestExecutable;
    double currentScore = scorePath(current);
    double bestScore = currentScore;
    double bestExecutableScore = numeric_limits<double>::infinity();

    if (firstMovesAreLegal(current)) {
        bestExecutable = current;
        bestExecutableScore = currentScore;
    }

    uniform_real_distribution<double> realPick(0.0, 1.0);
    for (
        double temperature = config.initialTemperature;
        temperature > config.minimumTemperature;
        temperature *= config.coolingRate
    ) {
        for (int i = 0; i < config.innerLoop; ++i) {
            vector<Point> candidate = mutatePath(current);
            double candidateScore = scorePath(candidate);
            double delta = currentScore - candidateScore;
            if (delta > 0.0 || exp(delta / temperature) > realPick(rng)) {
                current = std::move(candidate);
                currentScore = candidateScore;
                if (currentScore < bestScore) {
                    best = current;
                    bestScore = currentScore;
                }
                if (firstMovesAreLegal(current) && currentScore < bestExecutableScore) {
                    bestExecutable = current;
                    bestExecutableScore = currentScore;
                }
            }
        }
    }

    if (!bestExecutable.empty()) {
        return bestExecutable;
    }
    return best;
}

vector<Point> legalBfsPath(const Point& start, const Point& target) {
    bool visited[MAP_SIZE][MAP_SIZE] = {};
    Point parent[MAP_SIZE][MAP_SIZE];
    queue<Point> frontier;
    frontier.push(start);
    visited[start.x][start.y] = true;

    while (!frontier.empty()) {
        Point current = frontier.front();
        frontier.pop();
        if (samePoint(current, target)) {
            break;
        }
        vector<Point> candidates = nextPoints(current);
        sort(candidates.begin(), candidates.end(), [&](const Point& a, const Point& b) {
            return torusDistance(a, target) < torusDistance(b, target);
        });
        for (const Point& next : candidates) {
            if (!available(next) || visited[next.x][next.y]) {
                continue;
            }
            visited[next.x][next.y] = true;
            parent[next.x][next.y] = current;
            frontier.push(next);
        }
    }

    if (!visited[target.x][target.y]) {
        return {start};
    }

    vector<Point> reversed;
    Point current = target;
    while (!samePoint(current, start)) {
        reversed.push_back(current);
        current = parent[current.x][current.y];
    }
    reverse(reversed.begin(), reversed.end());
    vector<Point> path{start};
    path.insert(path.end(), reversed.begin(), reversed.end());
    return path;
}

int countWallVisits(const vector<Point>& path) {
    int count = 0;
    for (int i = 1; i < static_cast<int>(path.size()); ++i) {
        if (!available(path[i])) {
            ++count;
        }
    }
    return count;
}

string requireValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw invalid_argument(string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

void parseArguments(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        string argument = argv[i];
        if (argument == "--seed") {
            config.seeded = true;
            config.seed = static_cast<uint32_t>(stoul(requireValue(argc, argv, i)));
        } else if (argument == "--map-file") {
            config.mapFile = requireValue(argc, argv, i);
        } else if (argument == "--inner-loop") {
            config.innerLoop = stoi(requireValue(argc, argv, i));
        } else if (argument == "--initial-temp") {
            config.initialTemperature = stod(requireValue(argc, argv, i));
        } else if (argument == "--min-temp") {
            config.minimumTemperature = stod(requireValue(argc, argv, i));
        } else if (argument == "--cooling") {
            config.coolingRate = stod(requireValue(argc, argv, i));
        } else if (argument == "--move-distance-weight") {
            config.moveDistanceWeight = stod(requireValue(argc, argv, i));
        } else if (argument == "--wall-penalty") {
            config.wallPenalty = stod(requireValue(argc, argv, i));
        } else if (argument == "--early-wall-penalty") {
            config.earlyWallPenalty = stod(requireValue(argc, argv, i));
        } else if (argument == "--greedy-steps") {
            config.greedySteps = stoi(requireValue(argc, argv, i));
        } else if (argument == "--segment-span") {
            config.segmentSpan = stoi(requireValue(argc, argv, i));
        } else if (argument == "--early-bias") {
            config.earlyBias = stod(requireValue(argc, argv, i));
        } else {
            throw invalid_argument("unknown argument: " + argument);
        }
    }

    if (config.innerLoop <= 0 || config.initialTemperature <= config.minimumTemperature ||
        config.minimumTemperature <= 0.0 || config.coolingRate <= 0.0 ||
        config.coolingRate >= 1.0 || config.moveDistanceWeight <= 0.0 ||
        config.wallPenalty < 0.0 ||
        config.earlyWallPenalty < config.wallPenalty || config.greedySteps < 0 ||
        config.segmentSpan <= 0 || config.earlyBias < 0.0 || config.earlyBias > 1.0) {
        throw invalid_argument("invalid experimental SA parameter");
    }
}

void readMap() {
    ifstream input(config.mapFile);
    if (!input) {
        throw runtime_error("cannot open map file: " + config.mapFile);
    }

    int cell = 0;
    for (int row = 0; row < MAP_SIZE; ++row) {
        for (int col = 0; col < MAP_SIZE; ++col) {
            if (!(input >> cell)) {
                throw runtime_error("map file is incomplete");
            }
            walls[row][col] = cell != 0;
        }
    }
    if (!(input >> humanX >> humanY >> monsterX >> monsterY)) {
        throw runtime_error("map file is missing spawn coordinates");
    }

    int previousMoveFlag = 0;
    if (input >> previousMoveFlag) {
        if (
            previousMoveFlag == 1 &&
            input >> previousMoveFrom.x >> previousMoveFrom.y
                  >> previousMoveTo.x >> previousMoveTo.y
        ) {
            previousMoveFrom = {wrap(previousMoveFrom.x), wrap(previousMoveFrom.y)};
            previousMoveTo = {wrap(previousMoveTo.x), wrap(previousMoveTo.y)};
            hasPreviousMove = true;
        }
    }
}

void writeMap(const Point& movedMonster) {
    ofstream output(config.mapFile);
    if (!output) {
        throw runtime_error("cannot write map file: " + config.mapFile);
    }
    for (int row = 0; row < MAP_SIZE; ++row) {
        for (int col = 0; col < MAP_SIZE; ++col) {
            output << walls[row][col] << ' ';
        }
        output << '\n';
    }
    output << humanX << ' ' << humanY << '\n';
    output << movedMonster.x << ' ' << movedMonster.y;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        parseArguments(argc, argv);
        rng.seed(config.seeded ? config.seed : random_device{}());
        readMap();

        Point start{monsterX, monsterY};
        Point target{humanX, humanY};
        vector<Point> initialPath = randomShortestPath(start, target);
        vector<Point> selectedPath = simulatedAnnealing(initialPath);

        bool fallbackUsed = !firstMovesAreLegal(selectedPath);
        if (fallbackUsed) {
            selectedPath = legalBfsPath(start, target);
        }
        Point movedMonster = actualMovePoint(selectedPath);
        if (!available(movedMonster)) {
            selectedPath = {start};
            movedMonster = start;
            fallbackUsed = true;
        }

        writeMap(movedMonster);
        cerr << "SA_STATS"
             << " fallback=" << (fallbackUsed ? 1 : 0)
             << " path_length=" << selectedPath.size()
             << " wall_visits=" << countWallVisits(selectedPath)
             << " score=" << scorePath(selectedPath)
             << '\n';
        return 0;
    } catch (const exception& error) {
        cerr << "experimental SA failed: " << error.what() << '\n';
        return 1;
    }
}
