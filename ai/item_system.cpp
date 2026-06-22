#include "Item_system.h"

#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <queue>
#include <limits>

using namespace std;

static mt19937 rng(random_device{}());


// ===============================
// Utility Functions
// ===============================

int randomInt(int low, int high) {
    uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}

bool samePosition(Position a, Position b) {
    return a.row == b.row && a.col == b.col;
}

string itemName(ItemType type) {
    if (type == SPEED_BOOTS) {
        return "Speed Boots";
    }
    else if (type == HOME_STONE) {
        return "Safe Teleport Stone";
    }
    else if (type == FREEZE_TRAP) {
        return "Freeze Trap";
    }
    else {
        return "Invisibility Cloak";
    }
}

char itemSymbol(ItemType type) {
    if (type == SPEED_BOOTS) {
        return 'S';
    }
    else if (type == HOME_STONE) {
        return 'T';
    }
    else if (type == FREEZE_TRAP) {
        return 'F';
    }
    else {
        return 'C';
    }
}


// ===============================
// Map and Coordinate Functions
// ===============================

Position wrapPosition(Position pos, const vector<vector<int>>& grid) {
    int rows = (int)grid.size();
    int cols = (int)grid[0].size();

    Position wrapped;
    wrapped.row = ((pos.row % rows) + rows) % rows;
    wrapped.col = ((pos.col % cols) + cols) % cols;

    return wrapped;
}

int torusDistance(Position a, Position b, const vector<vector<int>>& grid) {
    int rows = (int)grid.size();
    int cols = (int)grid[0].size();

    int rowDiff = abs(a.row - b.row);
    int colDiff = abs(a.col - b.col);

    int wrappedRowDiff = min(rowDiff, rows - rowDiff);
    int wrappedColDiff = min(colDiff, cols - colDiff);

    return wrappedRowDiff + wrappedColDiff;
}

bool isWalkable(const vector<vector<int>>& grid, Position pos) {
    Position p = wrapPosition(pos, grid);
    return grid[p.row][p.col] == FLOOR;
}

bool isMonsterHere(Position pos, const vector<MonsterState>& monsters) {
    for (const MonsterState& monster : monsters) {
        if (samePosition(pos, monster.pos)) {
            return true;
        }
    }
    return false;
}

bool isItemHere(Position pos, const vector<Item>& items) {
    for (const Item& item : items) {
        if (item.active && samePosition(pos, item.pos)) {
            return true;
        }
    }
    return false;
}

bool canSpawnItem(
    const vector<vector<int>>& grid,
    Position pos,
    const PlayerState& player,
    const vector<MonsterState>& monsters,
    const vector<Item>& items,
    const vector<Item>& traps
) {
    pos = wrapPosition(pos, grid);

    if (!isWalkable(grid, pos)) return false;
    if (samePosition(pos, player.pos)) return false;
    if (isMonsterHere(pos, monsters)) return false;
    if (isItemHere(pos, items)) return false;
    if (isItemHere(pos, traps)) return false;

    return true;
}

vector<Position> getWalkableNeighbors(
    Position pos,
    const vector<vector<int>>& grid
) {
    vector<Position> result;
    vector<pair<int, int>> moves = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };

    for (auto move : moves) {
        Position nextPos;
        nextPos.row = pos.row + move.first;
        nextPos.col = pos.col + move.second;
        nextPos = wrapPosition(nextPos, grid);

        if (isWalkable(grid, nextPos)) {
            result.push_back(nextPos);
        }
    }
    return result;
}


// ===============================
// Core Algorithm: Distance Field
// ===============================

static vector<vector<int>> computeDistanceField(
    const vector<vector<int>>& grid, 
    const vector<Position>& sources
) {
    int rows = (int)grid.size();
    int cols = (int)grid[0].size();
    
    const int INF = numeric_limits<int>::max();
    vector<vector<int>> distField(rows, vector<int>(cols, INF));
    queue<Position> q;
    
    for (const Position& src : sources) {
        distField[src.row][src.col] = 0;
        q.push(src);
    }
    
    while (!q.empty()) {
        Position cur = q.front();
        q.pop();
        
        int currentDist = distField[cur.row][cur.col];
        vector<Position> neighbors = getWalkableNeighbors(cur, grid);
        
        for (Position nxt : neighbors) {
            if (distField[nxt.row][nxt.col] > currentDist + 1) {
                distField[nxt.row][nxt.col] = currentDist + 1;
                q.push(nxt);
            }
        }
    }
    return distField;
}


// ===============================
// Safe Teleport Function
// ===============================

Position findSafeTeleportPosition(
    const vector<vector<int>>& grid,
    const PlayerState& player,
    const vector<MonsterState>& monsters,
    int /*minDistance*/,
    const vector<Item>* items,
    const vector<Item>* traps
) {
    int rows = (int)grid.size();
    int cols = (int)grid[0].size();

    vector<Position> monsterPositions;
    for (const MonsterState& m : monsters) {
        monsterPositions.push_back(m.pos);
    }
    vector<vector<int>> hazardField = computeDistanceField(grid, monsterPositions);

    vector<pair<int, Position>> candidates;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            Position candidate = {r, c};

            if (!isWalkable(grid, candidate) || samePosition(candidate, player.pos)
                || isMonsterHere(candidate, monsters)
                || (items != nullptr && isItemHere(candidate, *items))
                || (traps != nullptr && isItemHere(candidate, *traps))) {
                continue;
            }
            candidates.push_back({hazardField[r][c], candidate});
        }
    }
    if (candidates.empty()) return player.pos;
    sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    size_t safestCount = max<size_t>(1, (candidates.size() + 9) / 10);
    return candidates[randomInt(0, static_cast<int>(safestCount) - 1)].second;
}


// ===============================
// Item Spawning Functions
// ===============================

ItemType randomItemType(bool cloakAlreadySpawned) {
    int totalWeight = cloakAlreadySpawned ? 90 : 95;
    int roll = randomInt(1, totalWeight);

    if (!cloakAlreadySpawned && roll <= 5) return INVISIBILITY_CLOAK;

    int currentOffset = cloakAlreadySpawned ? 0 : 5;

    if (roll <= currentOffset + 20) return HOME_STONE;
    else if (roll <= currentOffset + 20 + 35) return FREEZE_TRAP;
    else return SPEED_BOOTS;
}

void addFixedItem(vector<Item>& items, ItemType type, Position pos) {
    Item item;
    item.type = type;
    item.pos = pos;
    item.active = true;
    item.lifetime = (type == FREEZE_TRAP) ? FREEZE_TRAP_LIFETIME : -1;
    items.push_back(item);
}

void spawnRandomItems(
    vector<Item>& items,
    const vector<vector<int>>& grid,
    const PlayerState& player,
    const vector<MonsterState>& monsters,
    const vector<Item>& traps,
    bool& cloakAlreadySpawned,
    int itemCount
) {
    int rows = (int)grid.size();
    int cols = (int)grid[0].size();

    int spawned = 0;
    int attempts = 0;

    while (spawned < itemCount && attempts < 500) {
        attempts++;

        Position pos;
        pos.row = randomInt(0, rows - 1);
        pos.col = randomInt(0, cols - 1);
        pos = wrapPosition(pos, grid);

        if (!canSpawnItem(grid, pos, player, monsters, items, traps)) {
            continue;
        }

        ItemType type = randomItemType(cloakAlreadySpawned);

        Item item;
        item.type = type;
        item.pos = pos;
        item.active = true;
        item.lifetime = (type == FREEZE_TRAP) ? FREEZE_TRAP_LIFETIME : -1;

        if (type == INVISIBILITY_CLOAK) cloakAlreadySpawned = true;

        items.push_back(item);
        spawned++;

        cout << "[UI Info] Item spawned: " << itemName(type) 
             << " at (" << pos.row << ", " << pos.col << ")" << endl;
    }
}


// ===============================
// Item Effect and State Update Functions
// ===============================

void updateTrapLifetimeByStep(vector<Item>& traps) {
    for (Item& trap : traps) {
        if (trap.active && trap.lifetime > 0) {
            trap.lifetime--;
            if (trap.lifetime <= 0) {
                trap.active = false;
                cout << "[UI Info] Freeze Trap lifetime expired. It disappeared." << endl;
            }
        }
    }
}

void decayStatesAfterSuccessfulPlayerStep(
    PlayerState& player,
    vector<MonsterState>& monsters,
    vector<Item>& traps
) {
    updateTrapLifetimeByStep(traps);

    if (player.speedBootsTurns > 0) {
        player.speedBootsTurns--;
    }

    if (player.invisibleTurns > 0) {
        player.invisibleTurns--;
    }

    for (MonsterState& monster : monsters) {
        if (monster.frozenTurns > 0) {
            monster.frozenTurns--;
        }
    }
}

static void finishSuccessfulPlayerInput(
    PlayerState& player,
    vector<MonsterState>& monsters,
    vector<Item>& traps,
    bool pickedBoots,
    bool pickedCloak,
    bool pickedFreeze
) {
    updateTrapLifetimeByStep(traps);

    if (!pickedBoots && player.speedBootsTurns > 0) {
        player.speedBootsTurns--;
    }
    if (!pickedCloak && player.invisibleTurns > 0) {
        player.invisibleTurns--;
    }
    if (!pickedFreeze) {
        for (MonsterState& monster : monsters) {
            if (monster.frozenTurns > 0) {
                monster.frozenTurns--;
            }
        }
    }
}

bool applyItemEffect(
    Item& item,
    PlayerState& player,
    vector<Item>& traps,
    bool& cloakAlreadySpawned,
    const vector<vector<int>>& grid,
    vector<MonsterState>& monsters,
    const vector<Item>* allItems
) {
    if (item.type == SPEED_BOOTS) {
        player.speedBootsTurns = SPEED_BOOTS_DURATION;
        cout << "[Effect] Speed Boots active for " << SPEED_BOOTS_DURATION
             << " movement inputs." << endl;
        item.active = false;
        return false;
    }
    else if (item.type == HOME_STONE) {
        Position newPos = findSafeTeleportPosition(
            grid, player, monsters, TELEPORT_SAFE_DISTANCE, allItems, &traps
        );
        player.pos = newPos;
        cout << "[Effect] Safe Teleport Stone triggered: Player teleported to safe position ("
             << player.pos.row << ", " << player.pos.col << "). Movement phase aborted." << endl;
        item.active = false;
        return true;
    }
    else if (item.type == FREEZE_TRAP) {
        for (MonsterState& monster : monsters) {
            monster.frozenTurns = max(
                monster.frozenTurns,
                FREEZE_TRAP_DURATION
            );
        }
        cout << "[Effect] Freeze Trap triggered: Monsters frozen for "
             << FREEZE_TRAP_DURATION << " player steps." << endl;
        item.active = false;
        return false;
    }
    else if (item.type == INVISIBILITY_CLOAK) {
        player.invisibleTurns = INVISIBILITY_DURATION;
        cloakAlreadySpawned = true;
        cout << "[Effect] Invisibility Cloak triggered: Player invisible for "
             << INVISIBILITY_DURATION << " player steps." << endl;
        item.active = false;
        return false;
    }
    return false;
}

bool checkPlayerItemPickup(
    PlayerState& player,
    vector<Item>& items,
    vector<Item>& traps,
    bool& cloakAlreadySpawned,
    const vector<vector<int>>& grid,
    vector<MonsterState>& monsters
) {
    for (Item& item : items) {
        if (item.active && samePosition(player.pos, item.pos)) {
            return applyItemEffect(
                item, player, traps, cloakAlreadySpawned, grid, monsters, &items
            );
        }
    }
    return false;
}

void checkMonsterTrap(vector<MonsterState>& monsters, vector<Item>& traps) {
    for (int i = 0; i < (int)monsters.size(); i++) {
        for (Item& trap : traps) {
            if (trap.active && samePosition(monsters[i].pos, trap.pos)) {
                monsters[i].frozenTurns = max(
                    monsters[i].frozenTurns,
                    FREEZE_TRAP_DURATION
                );
                trap.active = false;
                cout << "[Combat] Monster " << i + 1 
                     << " stepped on Freeze Trap! Frozen for "
                     << FREEZE_TRAP_DURATION << " player steps." << endl;
            }
        }
    }
}


// ===============================
// Movement and Game State Functions
// ===============================

static void moveMonsterOneStepGreedy(
    MonsterState& monster,
    const vector<vector<int>>& grid,
    const vector<vector<int>>& playerDistField
) {
    if (monster.frozenTurns > 0) return;

    int bestDist = playerDistField[monster.pos.row][monster.pos.col];
    Position bestPos = monster.pos;

    vector<Position> neighbors = getWalkableNeighbors(monster.pos, grid);
    
    for (Position nxt : neighbors) {
        if (playerDistField[nxt.row][nxt.col] < bestDist) {
            bestDist = playerDistField[nxt.row][nxt.col];
            bestPos = nxt;
        }
    }

    monster.pos = bestPos;
}

bool moveMonsters(
    vector<MonsterState>& monsters,
    const PlayerState& player,
    const vector<vector<int>>& grid,
    vector<Item>& traps
) {
    const bool invisible = player.invisibleTurns > 0;

    vector<vector<int>> playerDistField;
    if (!invisible) {
        vector<Position> playerPosList = {player.pos};
        playerDistField = computeDistanceField(grid, playerPosList);
    }

    for (int i = 0; i < (int)monsters.size(); i++) {
        for (int step = 0; step < 2; step++) {

            if (monsters[i].frozenTurns > 0) {
                break;
            }
            else if (invisible) {
                vector<Position> neighbors = getWalkableNeighbors(monsters[i].pos, grid);
                vector<Position> preferred;
                for (Position pos : neighbors) {
                    if (!samePosition(pos, player.pos)) preferred.push_back(pos);
                }
                const vector<Position>& choices = preferred.empty() ? neighbors : preferred;
                if (!choices.empty()) {
                    monsters[i].pos = choices[randomInt(0, static_cast<int>(choices.size()) - 1)];
                }
            } else {
                moveMonsterOneStepGreedy(monsters[i], grid, playerDistField);
            }
            checkMonsterTrap(monsters, traps);

            if (!invisible && samePosition(monsters[i].pos, player.pos)) {
                cout << "[Game Over] Monster " << i + 1 
                     << " caught the player during chase." << endl;
                return true;
            }

            if (monsters[i].frozenTurns > 0) {
                break;
            }
        }
    }

    return false;
}

bool movePlayerOneStep(
    PlayerState& player,
    int dr,
    int dc,
    const vector<vector<int>>& grid
) {
    Position nextPos;
    nextPos.row = player.pos.row + dr;
    nextPos.col = player.pos.col + dc;
    nextPos = wrapPosition(nextPos, grid);

    if (isWalkable(grid, nextPos)) {
        player.pos = nextPos;
        return true;
    }

    return false;
}

MoveResult movePlayerWithItemCheck(
    PlayerState& player,
    int dr,
    int dc,
    const vector<vector<int>>& grid,
    vector<Item>& items,
    vector<Item>& traps,
    bool& cloakAlreadySpawned,
    vector<MonsterState>& monsters,
    bool dashRequested
) {
    const int maxSteps = player.speedBootsTurns <= 0 ? 1
        : (dashRequested ? SPEED_BOOTS_DASH_STEPS : SPEED_BOOTS_NORMAL_STEPS);
    bool moved = false;
    bool pickedBoots = false;
    bool pickedCloak = false;
    bool pickedFreeze = false;

    for (int step = 0; step < maxSteps; ++step) {
        if (!movePlayerOneStep(player, dr, dc, grid)) break;
        moved = true;
        if (checkGameOver(player, monsters)) return MOVE_CAUGHT;

        for (Item& item : items) {
            if (!item.active || !samePosition(player.pos, item.pos)) continue;
            pickedBoots = pickedBoots || item.type == SPEED_BOOTS;
            pickedCloak = pickedCloak || item.type == INVISIBILITY_CLOAK;
            pickedFreeze = pickedFreeze || item.type == FREEZE_TRAP;
            bool stopMovement = applyItemEffect(
                item, player, traps, cloakAlreadySpawned, grid, monsters, &items
            );
            if (stopMovement) {
                finishSuccessfulPlayerInput(
                    player,
                    monsters,
                    traps,
                    pickedBoots,
                    pickedCloak,
                    pickedFreeze
                );
                return MOVE_END_PHASE;
            }
            break;
        }
    }
    if (!moved) return MOVE_BLOCKED;

    finishSuccessfulPlayerInput(
        player,
        monsters,
        traps,
        pickedBoots,
        pickedCloak,
        pickedFreeze
    );

    return MOVE_CONTINUE;
}

bool checkGameOver(
    const PlayerState& player,
    const vector<MonsterState>& monsters
) {
    if (player.invisibleTurns > 0) return false;
    for (const MonsterState& monster : monsters) {
        if (samePosition(player.pos, monster.pos)) {
            return true;
        }
    }
    return false;
}
