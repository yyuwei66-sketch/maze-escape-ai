#pragma once

#include <vector>
#include <string>

// ===============================
// Basic Map Constants
// ===============================

inline constexpr int FLOOR = 0;
inline constexpr int WALL = 1;

// ===============================
// Item Balance Constants
// ===============================

inline constexpr int SPEED_BOOTS_EXTRA_STEPS = 3;
inline constexpr int TELEPORT_SAFE_DISTANCE = 6;
inline constexpr int FREEZE_TRAP_LIFETIME = 20;
inline constexpr int FREEZE_TRAP_DURATION = 3;
inline constexpr int INVISIBILITY_DURATION = 4;

// ===============================
// Basic Data Structures
// ===============================

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

enum MoveResult {
    MOVE_BLOCKED,
    MOVE_CONTINUE,
    MOVE_CAUGHT,
    MOVE_END_PHASE
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

// ===============================
// Utility Functions
// ===============================

int randomInt(int low, int high);

bool samePosition(Position a, Position b);

std::string itemName(ItemType type);

char itemSymbol(ItemType type);

// ===============================
// Map and Coordinate Functions
// ===============================

Position wrapPosition(
    Position pos,
    const std::vector<std::vector<int>>& grid
);

int torusDistance(
    Position a,
    Position b,
    const std::vector<std::vector<int>>& grid
);

bool isWalkable(
    const std::vector<std::vector<int>>& grid,
    Position pos
);

bool isMonsterHere(
    Position pos,
    const std::vector<MonsterState>& monsters
);

bool isItemHere(
    Position pos,
    const std::vector<Item>& items
);

bool canSpawnItem(
    const std::vector<std::vector<int>>& grid,
    Position pos,
    const PlayerState& player,
    const std::vector<MonsterState>& monsters,
    const std::vector<Item>& items,
    const std::vector<Item>& traps
);

std::vector<Position> getWalkableNeighbors(
    Position pos,
    const std::vector<std::vector<int>>& grid
);

// ===============================
// Safe Teleport Function
// ===============================

Position findSafeTeleportPosition(
    const std::vector<std::vector<int>>& grid,
    const PlayerState& player,
    const std::vector<MonsterState>& monsters,
    int minDistance = TELEPORT_SAFE_DISTANCE
);

// ===============================
// Item Spawning Functions
// ===============================

ItemType randomItemType(
    bool cloakAlreadySpawned
);

void addFixedItem(
    std::vector<Item>& items,
    ItemType type,
    Position pos
);

void spawnRandomItems(
    std::vector<Item>& items,
    const std::vector<std::vector<int>>& grid,
    const PlayerState& player,
    const std::vector<MonsterState>& monsters,
    const std::vector<Item>& traps,
    bool& cloakAlreadySpawned,
    int itemCount = 2
);

// ===============================
// Item Effect and State Update Functions
// ===============================

void updateTrapLifetimeByStep(
    std::vector<Item>& traps
);

void decayStatesAfterSuccessfulPlayerStep(
    PlayerState& player,
    std::vector<MonsterState>& monsters,
    std::vector<Item>& traps
);

bool applyItemEffect(
    Item& item,
    PlayerState& player,
    std::vector<Item>& traps,
    bool& cloakAlreadySpawned,
    const std::vector<std::vector<int>>& grid,
    std::vector<MonsterState>& monsters
);

bool checkPlayerItemPickup(
    PlayerState& player,
    std::vector<Item>& items,
    std::vector<Item>& traps,
    bool& cloakAlreadySpawned,
    const std::vector<std::vector<int>>& grid,
    std::vector<MonsterState>& monsters
);

void checkMonsterTrap(
    std::vector<MonsterState>& monsters,
    std::vector<Item>& traps
);

// ===============================
// Movement and Game State Functions
// ===============================

bool moveMonsters(
    std::vector<MonsterState>& monsters,
    const PlayerState& player,
    const std::vector<std::vector<int>>& grid,
    std::vector<Item>& traps
);

bool movePlayerOneStep(
    PlayerState& player,
    int dr,
    int dc,
    const std::vector<std::vector<int>>& grid
);

MoveResult movePlayerWithItemCheck(
    PlayerState& player,
    int dr,
    int dc,
    const std::vector<std::vector<int>>& grid,
    std::vector<Item>& items,
    std::vector<Item>& traps,
    bool& cloakAlreadySpawned,
    std::vector<MonsterState>& monsters
);

bool checkGameOver(
    const PlayerState& player,
    const std::vector<MonsterState>& monsters
);
