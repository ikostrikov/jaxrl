#!/bin/bash

find . -name '*.py' -print0 | xargs -0 isort
find . -name '*.py' -print0 | xargs -0 yapf -i