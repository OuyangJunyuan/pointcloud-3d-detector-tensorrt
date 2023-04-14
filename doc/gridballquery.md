# Comparison of BallQuery and GridBallQuery

|               |      BallQuery      |   GridBallQuery   |
|:--------------|:-------------------:|:-----------------:|
| Complexsity   | $\mathcal{O}(NK^3)$ | $\mathcal{O}(NM)$ |
| Runtime       |       0.04 ms       |      2.5 ms       |
| GPU occupancy |         30%         |        55%        |

# Implementation

```python

parallely for i,src in enumerate(srcs):
    src_voxel_coord = src / voxel
    for dx in [-k, k]:
        for dy in [-k, k]:
            for dz in [-k, k]:
                query_voxel_coord = v+[dx,dy,dz]
                if not voxel_hash(query_voxel_coord): break
                j = query_id(query_voxel_coord)
                if norm(src-queries[j]) > radius: break
                cnt[i]+=1
                ind[i,cnt[i]]=j
```