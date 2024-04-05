from pivotpoint import *
import random

def create_pivots_data(path: str, gapset: list[float], nsamples: int, value_range: tuple[float,float], radius_range: tuple[float,float]):
    data = [pivot_from_params(
                random.uniform(value_range[0],value_range[1]),
                gapset[random.randint(0,len(gapset) - 1)],
                random.uniform(radius_range[0],radius_range[1]))
            for _ in range(nsamples)]
    io.write_pivots_to_csv(path,data)

def create_clusters_data():
    pass

def create_samples_data():
    pass

path = "./test/input/pivots_test.txt"
gaps = [1/2, 1/3, 1/7]
nsamples = 10
value_range = [2,20]
radius_range = [1 + min(value_range) + max(gaps),5]
create_pivots_data(path,gaps,nsamples,value_range,radius_range)