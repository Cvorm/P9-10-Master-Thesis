import sys
import math

def sigmoid(x):
    print(f'Result: {1 / (1 + math.exp(-x))}')

inp = sys.argv
sigmoid(float(inp[1]))
