#include "ItemSystem.h"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

mt19937 rng(random_device{}());

int randomInt(int low, int high) {
    uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}

bool samePosition(Position a, Position b) {
    return a.row == b.row && a.col == b.col;
}

string itemName(ItemType type) {
    if (type == SPEED_BOOTS) return "Speed Boots";
    if (type == HOME_STONE) return "Home Stone";
    if (type == FREEZE_TRAP) return "Freeze Trap";
    return "Invisibility Cloak";
}

char itemSymbol(ItemType type) {
    if (type == SPEED_BOOTS) return 'S';
    if (type == HOME_STONE) return 'T';
    if (type == FREEZE_TRAP) return 'F';
    return 'C';
}

Position wrapPosition(Position pos, const vector<vector<int>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    Position wrapped;
    wrapped.row = (pos.row + rows) % rows;
    wrapped.col = (pos.col + cols) % cols;
    return wrapped;
}

int torusDistance(Position a, Position b, const vector<vector<int>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();

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
        if (samePosition(pos, monster.pos)) return true;
    }
    return false;
}

bool isItemHere(Position pos, const vector<Item>& items) {
    for (const Item& item : items) {
        if (item.active && samePosition(pos, item.pos)) return true;
    }
    return false;
}

bool canSpawnItem(
    const vector<vector<int>>& grid,
    Position pos,
    const PlayerState& player,
    const vector<MonsterState>& monsters,
    const vector<Item>& items
) {
    pos = wrapPosition(pos, grid);
    if (!isWalkable(grid, pos)) return false;
    if (samePosition(pos, player.pos)) return false;
    if (isMonsterHere(pos, monsters)) return false;
    if (isItemHere(pos, items)) return false;
    return true;
}

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
    item.lifetime = (type == FREEZE_TRAP) ? 15 : -1;
    items.push_back(item);
}

void spawnRandomItems(
    vector<Item>& items,
    const vector<vector<int>>& grid,
    const PlayerState& player,
    const vector<MonsterState>& monsters,
    bool& cloakAlreadySpawned,
    int itemCount
) {
    int rows = grid.size();
    int cols = grid[0].size();
    int spawned = 0;
    int attempts = 0;

    while (spawned < itemCount && attempts < 500) {
        attempts++;
        Position pos;
        pos.row = randomInt(0, rows - 1);
        pos.col = randomInt(0, cols - 1);

        if (!canSpawnItem(grid, pos, player, monsters, items)) continue;

        ItemType type = randomItemType(cloakAlreadySpawned);

        Item item;
        item.type = type;
        item.pos = pos;
        item.active = true;
        item.lifetime = (type == FREEZE_TRAP) ? 15 : -1;

        if (type == INVISIBILITY_CLOAK) cloakAlreadySpawned = true;

        items.push_back(item);
        spawned++;

        cout << "[UI Info] Item spawned: " << itemName(type) << " at (" << pos.row << ", " << pos.col << ")" << endl;
    }
}

void updateTrapLifetimeByStep(vector<Item>& traps) {
    for (Item& trap : traps) {
        if (trap.active && trap.lifetime > 0) {
            trap.lifetime--;
            if (trap.lifetime <= 0) {
                trap.active = false;
                cout << "[UI Info] Freeze Trap lifetime (15 steps) expired. It disappeared." << endl;
            }
        }
    }
}

bool applyItemEffect(
    Item& item,
    PlayerState& player,
    vector<Item>& traps,
    bool& cloakAlreadySpawned
) {
    if (item.type == SPEED_BOOTS) {
        player.extraSteps += 2;
        cout << "[Effect] Speed Boots triggered: Extra 2 steps granted immediately!" << endl;
        item.active = false;
        return false;
    }
    else if (item.type == HOME_STONE) {
        player.pos = player.spawnPos;
        cout << "[Effect] Home Stone triggered: Teleported back to spawn point. Movement phase aborted." << endl;
        item.active = false;
        player.extraSteps = 0;
        return true; 
    }
    else if (item.type == FREEZE_TRAP) {
        Item trap;
        trap.type = FREEZE_TRAP;
        trap.pos = item.pos; 
        trap.lifetime = 15;
        trap.active = true;
        traps.push_back(trap);
        cout << "[Effect] Freeze Trap triggered: Deployed at current cell. Active after player leaves." << endl;
        item.active = false;
        return false;
    }
    else if (item.type == INVISIBILITY_CLOAK) {
        player.invisibleTurns = 3;
        cloakAlreadySpawned = true;
        cout << "[Effect] Invisibility Cloak triggered: Equipped! Monsters lose target for 3 steps." << endl;
        item.active = false;
        return false;
    }
    return false;
}

bool checkPlayerItemPickup(
    PlayerState& player,
    vector<Item>& items,
    vector<Item>& traps,
    bool& cloakAlreadySpawned
) {
    for (Item& item : items) {
        if (item.active && samePosition(player.pos, item.pos)) {
            return applyItemEffect(item, player, traps, cloakAlreadySpawned);
        }
    }
    return false;
}

void checkMonsterTrap(vector<MonsterState>& monsters, vector<Item>& traps) {
    for (int i = 0; i < (int)monsters.size(); i++) {
        for (Item& trap : traps) {
            if (trap.active && samePosition(monsters[i].pos, trap.pos)) {
                monsters[i].frozenTurns = 2; 
                trap.active = false; 
                cout << "[Combat] Monster " << i + 1 << " stepped on Freeze Trap! Frozen for 2 player steps." << endl;
            }
        }
    }
}

void moveMonsterOneStep(
    MonsterState& monster, 
    const PlayerState& player, 
    const vector<vector<int>>& grid
) {
    if (monster.frozenTurns > 0) return; 
    if (player.invisibleTurns > 0) return; 
    
    if (samePosition(monster.pos, player.pos)) return;

    int bestDist = 1000000;
    Position bestPos = monster.pos;

    vector<pair<int, int>> moves = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
    for (auto move : moves) {
        Position nextPos;
        nextPos.row = monster.pos.row + move.first;
        nextPos.col = monster.pos.col + move.second;
        nextPos = wrapPosition(nextPos, grid);

        if (!isWalkable(grid, nextPos)) continue;

        // Apply toroidal distance for correct wrap-around greedy pathing
        int d = torusDistance(nextPos, player.pos, grid);
        if (d < bestDist) {
            bestDist = d;
            bestPos = nextPos;
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
    for (int i = 0; i < (int)monsters.size(); i++) {
        for (int step = 0; step < 2; step++) {
            moveMonsterOneStep(monsters[i], player, grid);
            checkMonsterTrap(monsters, traps);
            
            if (samePosition(monsters[i].pos, player.pos)) {
                cout << "[Game Over] Monster " << i + 1 << " caught the player during chase." << endl;
                return true;
            }
            
            if (monsters[i].frozenTurns > 0 || player.invisibleTurns > 0) {
                break;
            }
        }
    }
    return false;
}

bool movePlayerOneStep(PlayerState& player, int dr, int dc, const vector<vector<int>>& grid) {
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

void movePlayerWithItemCheck(
    PlayerState& player,
    int dr,
    int dc,
    const vector<vector<int>>& grid,
    vector<Item>& items,
    vector<Item>& traps,
    bool& cloakAlreadySpawned,
    vector<MonsterState>& monsters
) {
    int stepsTaken = 0;
    int maxSteps = 1;

    while (stepsTaken < maxSteps) {
        bool moved = movePlayerOneStep(player, dr, dc, grid);
        
        // Break movement loop immediately if hitting an un-walkable tile (e.g. wall)
        // This prevents state decay on invalid moves and cancels any remaining extra momentum
        if (!moved) {
            break; 
        }

        stepsTaken++;
        
        updateTrapLifetimeByStep(traps);
        
        if (player.invisibleTurns > 0) player.invisibleTurns--;
        
        for (MonsterState& monster : monsters) {
            if (monster.frozenTurns > 0) monster.frozenTurns--;
        }

        int beforeExtraSteps = player.extraSteps;
        bool stopMovement = checkPlayerItemPickup(player, items, traps, cloakAlreadySpawned);

        if (stopMovement) {
            player.extraSteps = 0;
            break; 
        }

        if (player.extraSteps > beforeExtraSteps) {
            maxSteps += player.extraSteps;
            player.extraSteps = 0; 
        }
    }
}

bool checkGameOver(const PlayerState& player, const vector<MonsterState>& monsters) {
    for (const MonsterState& monster : monsters) {
        if (samePosition(player.pos, monster.pos)) return true;
    }
    return false;
}