# --------------------------------------------------------------------------------------- #
# Plot the teacher agents. We can plot multiple log files one after the other, using
# built-in code from Spinup, or we can plot separately as shown below.
#
# Usually the NAME should have the algorithm after it, e.g., NAME=ant_td3, etc.
# --------------------------------------------------------------------------------------- #

#NAME=ant_td3
#python -m spinup.run plot data/${NAME}/${NAME}_s0 data/${NAME}/${NAME}_s10 data/${NAME}/${NAME}_s20  --count --savefig
#python -m spinup.run plot data/${NAME}/${NAME}_s0
#python -m spinup.run plot data/${NAME}/${NAME}_s10
#python -m spinup.run plot data/${NAME}/${NAME}_s20

SMOOTH=3
#HEAD=/data/spinup/data
HEAD=/data/spinup-data/mandi-spinup/data
# Before Jan 23, I was using primarily seed 10.
# for NAME in halfcheetah_td3_act0-1 halfcheetah_td3_act0-5 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s10  ${HEAD}/${NAME}/${NAME}_s20  \
#             --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done
#
# for NAME in hopper_td3_act0-1 hopper_td3_act0-5 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s10  ${HEAD}/${NAME}/${NAME}_s20  \
#         --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done
#
# # Special case here since Walker2d act=0.5 trained poorly.
# for NAME in walker2d_td3_act0-1 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s10  ${HEAD}/${NAME}/${NAME}_s20  \
#         --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done
# for NAME in walker2d_td3_act0-5 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s10  ${HEAD}/${NAME}/${NAME}_s20  ${HEAD}/${NAME}/${NAME}_s30  \
#         --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done


# Staring Jan 23, let's try using teacher seed 40, except Walekr2d uses 50.
# for NAME in halfcheetah_td3_act0-1 halfcheetah_td3_act0-5 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s40  \
#             --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done

# for NAME in hopper_td3_act0-1 hopper_td3_act0-5 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s40  \
#         --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done

# for NAME in walker2d_td3_act0-1 walker2d_td3_act0-5 ; do
#     python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s50  \
#         --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
# done
for NAME in halfcheetah_sac_act0-0 ; do
    python -m spinup.run plot  ${HEAD}/${NAME}/${NAME}_s40  \
            --title ${NAME}  --count  --savefig  --smooth ${SMOOTH}
done
