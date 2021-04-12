#!/bin/bash

BUILD=build

if [ ! -d "$BUILD" ]
then
  mkdir "$BUILD"
fi

git clone https://github.com/maximzubkov/positional-bias.git "$BUILD"/positional-bias
cd "$BUILD"/positional-bias && pip install -e .