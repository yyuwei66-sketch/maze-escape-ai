# Maze Escape: Human vs Monster AI Survival Game

## Project Overview

Maze Escape is an AI-based survival chase game developed for the SOF106 Principles of Artificial Intelligence final group project. The game is played on a 30x30 wrap-around maze map. The human moves one step per turn, while the monster moves two steps per turn.

The main purpose of this project is to compare different AI search and decision-making algorithms in a dynamic game environment.

## Main Features

- 30x30 grid-based maze map
- Wrap-around boundary mechanism
- Random spawn system
- Human moves 1 step per turn
- Monster moves 2 steps per turn
- Item generation every 10 human moves
- Unused items remain on the map
- Multiple monster AI algorithms
- BFS-based human escape strategy
- Genetic Algorithm map generation
- Basic UI and game visualization

## AI Algorithms

### Monster AI

1. Greedy Search
2. Simulated Annealing
3. Minimax
4. A* Pathfinding
5. Two-Monster Mode

### Human AI

1. BFS Escape Strategy

### Map Generation

1. Genetic Algorithm-based map generation

## Project Structure

```text
maze-escape-ai/
├── main.py
├── config/
├── game/
├── ai/
├── ga/
├── ui/
├── assets/
├── data/
├── tests/
├── docs/
└── screenshots/
