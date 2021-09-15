# ---------------------------------------------------------------------------------------------------- #
# (Feb 01) copying over new constanteps buffers. In general, follow this pattern for going TO lowland.
# ---------------------------------------------------------------------------------------------------- #
KEY=nonaddunif_
HEAD=/data/spinup/data
TARG=/data/seita/spinup/data

scp -r ${HEAD}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/buffer/*${KEY}*  seita@lowland.cs.berkeley.edu:${TARG}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40/buffer/
scp -r ${HEAD}/halfcheetah_td3_act0-5/halfcheetah_td3_act0-5_s40/buffer/*${KEY}*  seita@lowland.cs.berkeley.edu:${TARG}/halfcheetah_td3_act0-5/halfcheetah_td3_act0-5_s40/buffer/

scp -r ${HEAD}/hopper_td3_act0-1/hopper_td3_act0-1_s40/buffer/*${KEY}*            seita@lowland.cs.berkeley.edu:${TARG}/hopper_td3_act0-1/hopper_td3_act0-1_s40/buffer/
scp -r ${HEAD}/hopper_td3_act0-5/hopper_td3_act0-5_s40/buffer/*${KEY}*            seita@lowland.cs.berkeley.edu:${TARG}/hopper_td3_act0-5/hopper_td3_act0-5_s40/buffer/

scp -r ${HEAD}/walker2d_td3_act0-1/walker2d_td3_act0-1_s50/buffer/*${KEY}*        seita@lowland.cs.berkeley.edu:${TARG}/walker2d_td3_act0-1/walker2d_td3_act0-1_s50/buffer/
scp -r ${HEAD}/walker2d_td3_act0-5/walker2d_td3_act0-5_s50/buffer/*${KEY}*        seita@lowland.cs.berkeley.edu:${TARG}/walker2d_td3_act0-5/walker2d_td3_act0-5_s50/buffer/
