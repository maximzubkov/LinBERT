#!/bin/bash

BUILD=build

if [ ! -d "$BUILD" ]
then
  mkdir "$BUILD"
fi

RUN git clone git@github.com:maximzubkov/positional-bias.git "$BUILD"/positional-bias
cd "$BUILD"/positional-bias && pip install -e .