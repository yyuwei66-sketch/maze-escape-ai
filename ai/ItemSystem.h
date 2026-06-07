#pragma once

#include <vector>
#include <string>

const int FLOOR = 0;
const int WALL = 1;

struct Position {
    int row;
    int col;
};

enum ItemType {
    SPEED_BOOTS,
    HOME_STONE,
    FREEZE_TRAP,
    INVISIBILITY_CLOAK
};

struct Item {
    ItemType type;
    Position pos;
    int lifetime; 
    bool active;
};

struct PlayerState {
    Position pos;
    Position spawnPos;
    int extraSteps;
    int invisibleTurns;
};

struct MonsterState {
    Position pos;
    int frozenTurns;
};

// Utility Interfaces
int randomInt(int low, int high);
bool samePosition(Position a, Position b);
std::string itemName(ItemType type);
char itemSymbol(ItemType type);

// Map and Coordinate Interfaces
Position wrapPosition(Position pos, const std::vector<std::vector<int>>& grid);
int torusDistance(Position a, Position b, const std::vector<std::vector<int>>& grid);
bool isWalkable(const std::vector<std::vector<int>>& grid, Position pos);
bool isMonsterHere(Position pos, const std::vector<MonsterState>& monsters);
bool isItemHere(Position pos, const std::vector<Item>& items);
bool canSpawnItem(
    const std::vector<std::vector<int>>& grid,
    Position pos,
    const PlayerState& player,
    const std::vector<MonsterState>& monsters,
    const std::vector<Item>& items
);

// Item Spawning Interfaces
ItemType randomItemType(bool cloakAlreadySpawned);
void addFixedItem(std::vector<Item>& items, ItemType type, Position pos);
void spawnRandomItems(
    std::vector<Item>& items,
    const std::vector<std::vector<int>>& grid,
    const PlayerState& player,
    const std::vector<MonsterState>& monsters,
    bool& cloakAlreadySpawned,
    int itemCount = 2
);

// Logic and Settlement Interfaces
void updateTrapLifetimeByStep(std::vector<Item>& traps);
bool applyItemEffect(Item& item, PlayerState& player, std::vector<Item>& traps, bool& cloakAlreadySpawned);
bool checkPlayerItemPickup(PlayerState& player, std::vector<Item>& items, std::vector<Item>& traps, bool& cloakAlreadySpawned);
void checkMonsterTrap(std::vector<MonsterState>& monsters, std::vector<Item>& traps);

// Movement and Game Over Interfaces
void moveMonsterOneStep(MonsterState& monster, const PlayerState& player, const std::vector<std::vector<int>>& grid);
bool moveMonsters(std::vector<MonsterState>& monsters, const PlayerState& player, const std::vector<std::vector<int>>& grid, std::vector<Item>& traps);
bool movePlayerOneStep(PlayerState& player, int dr, int dc, const std::vector<std::vector<int>>& grid);
void movePlayerWithItemCheck(
    PlayerState& player,
    int dr,
    int dc,
    const std::vector<std::vector<int>>& grid,
    std::vector<Item>& items,
    std::vector<Item>& traps,
    bool& cloakAlreadySpawned,
    std::vector<MonsterState>& monsters
);
bool checkGameOver(const PlayerState& player, const std::vector<MonsterState>& monsters);