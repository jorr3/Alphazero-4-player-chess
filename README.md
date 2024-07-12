# AlphaZero for Four-Player Chess

## Project Overview

This project is an experimental effort to apply the AlphaZero methodology to four-player chess, building upon the
open-source chess engine, 4pchess. It is currently under active development, focusing on integrating deep learning and
Monte Carlo Tree Search (MCTS) to navigate the complex dynamics of four-player chess. The goal is to develop a robust AI
that learns optimal strategies through self-play.

## Key Components

- **AlphaZero Algorithm**: Employs self-play and reinforcement learning to master four-player chess without reliance on
  human data or prior domain knowledge.
- **Monte Carlo Tree Search (MCTS)**: Utilized for strategic decision-making, allowing effective exploration and
  exploitation of the game space.
- **Deep Learning**: Leverages a Residual Neural Network (ResNet) architecture to evaluate game states and predict move
  probabilities.
- **Four-Player Chess Environment**: An extension of traditional chess that introduces a new layer of strategy with four
  participants divided into two teams.

## Engine

This project builds upon the [4pchess](https://github.com/obryanlouis/4pchess) engine, incorporating modifications to
support the AlphaZero framework, removing non-essential code, and implementing a Python binding instead of communicating
via the command line.

You can find the 4pchess repo [here](https://github.com/obryanlouis/4pchess).
