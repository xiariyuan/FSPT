import os
import numpy as np

ld = os.listdir('.')
for li in ld:
    if os.path.isdir(li):
        print('working on', li)
        o = np.load('%s/anno.npz' % li)
        d = dict(o)

        info_path_out = '%s/info.npz' % li
        np.savez_compressed(
            info_path_out,
            trajs_2d=d['trajs_2d'].shape,
            trajs_3d=d['trajs_3d'].shape,
            valids=d['valids'].shape,
            visibs=d['visibs'].shape,
            intrinsics=d['intrinsics'].shape,
            extrinsics=d['extrinsics'].shape,
        )
        print('saved %s' % info_path_out)

