#!/bin/bash

if [ "$1" = "1" ]; then
	fileX=$2
	fileY=$3
	learningRate=$4
	timeGap=$5

	py Q1.py $fileX $fileY $learningRate $timeGap
fi

if [ "$1" = "2" ]; then
	fileX=$2
	fileY=$3
	tau=$4

	py Q2.py $fileX $fileY $tau
fi

if [ "$1" = "3" ]; then
	fileX=$2
	fileY=$3

	py Q3.py $fileX $fileY
fi

if [ "$1" = "4" ]; then
	fileX=$2
	fileY=$3
	partNo=$4

	py Q4.py $fileX $fileY $partNo
fi

