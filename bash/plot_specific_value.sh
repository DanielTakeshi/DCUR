# --------------------------------------------------------------------------------------- #
# Shows how to use the built-in Spinup plotter to plot a specific value.
#
# Usually the NAME should have the algorithm after it, e.g., NAME=ant_td3, etc.
# --------------------------------------------------------------------------------------- #

SMOOTH=3
HEAD=/data/spinup/data

# Plot a specific value from training. For offline RL runs, we need to use TotalGradSteps
# as the x-value because we don't have TotalEnvInteracts recorded (for obvious reasons).
# Here we can plot the 'RExtra' values, which are the intrinsic rewards.

## TITLE=offline_uniform_0.01_0.50_np
## for Y in Performance AverageRExtra AverageSigma1 AverageSigma2 AverageQ1Vals AverageQ2Vals ; do
##     python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_uniform_0.01_0.50_np/  \
##         --title ${TITLE}  --count  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
## done

# Look at Q-values for Final Buffer. I don't imagine they'd be diverging. And especially not for the real training runs of TD3.
# Also we can plot multiple values for -y but it actually makes separate plots, easier to save individually?


## # Original Training Run; x=TotalEnvInteracts.
## TITLE=online_td3
## for Y in Performance AverageQ1Vals AverageQ2Vals ; do
##     python -m spinup.run plot ${HEAD}/halfcheetah_td3_act0-1 ${HEAD}/halfcheetah_td3_act0-5  \
##         --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalEnvInteracts  -y ${Y}
## done
##
## # Final Buffer Case; x=TotalGradSteps.
## TITLE=offline_finalb
## for Y in Performance AverageQ1Vals AverageQ2Vals ; do
##     python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_finalb  \
##         --title ${TITLE}  --count  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
## done
##
## # Other buffers; x=TotalGradSteps.
## TITLE=uniform_0.01_0.50
## for Y in Performance AverageQ1Vals AverageQ2Vals ; do
##     python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_unif_0.01_0.50  ${HEAD}/halfcheetah_td3_offline_uniform_0.01_0.50_np  \
##         --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
## done


## # [Jan 19] some investigation into performance.
## # Some results: https://github.com/CannyLab/spinningup/pull/8
## TITLE=uniformeps_0.0_1.25_0.5_nofilt
## for Y in Performance AverageQ1Vals AverageQ2Vals ; do
##     python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_uniformeps_0.0_1.25_0.5_nofilt_np_10.0  \
##         --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##
##     python -m spinup.run plot ${HEAD}/hopper_td3_offline_uniformeps_0.0_1.25_0.5_nofilt_np_10.0  \
##         --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##
##     python -m spinup.run plot ${HEAD}/walker2d_td3_offline_uniformeps_0.0_1.25_0.5_nofilt_np_10.0  \
##         --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
## done


# ------------------------------------------------------------------------------------------------------ #
# [Feb 16, 2021] Investigate Q-value divergence, for results in:
# Data type uniformeps  https://github.com/CannyLab/spinningup/pull/9
# Data type logged      https://github.com/CannyLab/spinningup/pull/11
# Data type nonaddunif  https://github.com/CannyLab/spinningup/pull/14
# Note: we can do AverageQ2Vals as well, but those curves will be almost identical to AverageQ1Vals.
# Teacher seeds: 40, and for Walker2d, 50. For slides:
# https://docs.google.com/presentation/d/1V-7tg0DthrQY9azzmxWLdctMeCOT8hup8WUjIyxOzh8/edit?usp=sharing
# ------------------------------------------------------------------------------------------------------ #

## # These are the teachers we have, using specific seeds now.
## TITLE=online_td3
## for Y in Performance AverageQ1Vals ; do
##     python -m spinup.run plot ${HEAD}/halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40  \
##                               --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalEnvInteracts  -y ${Y}
##     python -m spinup.run plot ${HEAD}/hopper_td3_act0-1/hopper_td3_act0-1_s40  \
##                               --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalEnvInteracts  -y ${Y}
##     python -m spinup.run plot ${HEAD}/walker2d_td3_act0-1/walker2d_td3_act0-1_s50  \
##                               --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalEnvInteracts  -y ${Y}
## done

## for TITLE in finalb finalb_np_10.0 concurrent concurrent_np_10.0 ; do
##     for Y in Performance AverageQ1Vals ; do
##         python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_${TITLE}/halfcheetah_td3_act0-1_s40_s10 \
##                                   ${HEAD}/halfcheetah_td3_offline_${TITLE}/halfcheetah_td3_act0-1_s40_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##         python -m spinup.run plot ${HEAD}/hopper_td3_offline_${TITLE}/hopper_td3_act0-1_s40_s10 \
##                                   ${HEAD}/hopper_td3_offline_${TITLE}/hopper_td3_act0-1_s40_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##         python -m spinup.run plot ${HEAD}/walker2d_td3_offline_${TITLE}/walker2d_td3_act0-1_s50_s10 \
##                                   ${HEAD}/walker2d_td3_offline_${TITLE}/walker2d_td3_act0-1_s50_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##     done
## done

## for TITLE in nonaddunif_0.0_0.5_filter nonaddunif_0.0_0.5_filter_np_1.0 nonaddunif_0.0_0.5_filter_np_10.0 nonaddunif_0.0_0.5_filter_np_50.0 ; do
##     for Y in Performance AverageQ1Vals ; do
##         python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_${TITLE}/halfcheetah_td3_act0-1_s40_s10 \
##                                   ${HEAD}/halfcheetah_td3_offline_${TITLE}/halfcheetah_td3_act0-1_s40_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##         python -m spinup.run plot ${HEAD}/hopper_td3_offline_${TITLE}/hopper_td3_act0-1_s40_s10 \
##                                   ${HEAD}/hopper_td3_offline_${TITLE}/hopper_td3_act0-1_s40_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##         python -m spinup.run plot ${HEAD}/walker2d_td3_offline_${TITLE}/walker2d_td3_act0-1_s50_s10 \
##                                   ${HEAD}/walker2d_td3_offline_${TITLE}/walker2d_td3_act0-1_s50_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##     done
## done

## for TITLE in nonaddunif_0.0_1.0_filter nonaddunif_0.0_1.0_filter_np_1.0 nonaddunif_0.0_1.0_filter_np_10.0 nonaddunif_0.0_1.0_filter_np_50.0 ; do
##     for Y in Performance AverageQ1Vals ; do
##         python -m spinup.run plot ${HEAD}/halfcheetah_td3_offline_${TITLE}/halfcheetah_td3_act0-1_s40_s10 \
##                                   ${HEAD}/halfcheetah_td3_offline_${TITLE}/halfcheetah_td3_act0-1_s40_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##         python -m spinup.run plot ${HEAD}/hopper_td3_offline_${TITLE}/hopper_td3_act0-1_s40_s10 \
##                                   ${HEAD}/hopper_td3_offline_${TITLE}/hopper_td3_act0-1_s40_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##         python -m spinup.run plot ${HEAD}/walker2d_td3_offline_${TITLE}/walker2d_td3_act0-1_s50_s10 \
##                                   ${HEAD}/walker2d_td3_offline_${TITLE}/walker2d_td3_act0-1_s50_s20 \
##                                   --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalGradSteps  -y ${Y}
##     done
## done


# ------------------------------------------------------------------------------------------------------ #
# [Feb 28, 2021] Plotting more rewards. https://github.com/CannyLab/spinningup/pull/21
# We're trying to find the right kind of scaling for the extra reward. These are for the teachers.
# ------------------------------------------------------------------------------------------------------ #
SMOOTH=6
HEAD=/data/spinup/data

for TITLE in halfcheetah_td3_act0-1 hopper_td3_act0-1 ; do
    for Y in AverageRew MaxRew MinRew ; do
        python -m spinup.run plot ${HEAD}/${TITLE}/${TITLE}_s40  --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalEnvInteracts  -y ${Y}
    done
done
for TITLE in ant_td3_act0-1 walker2d_td3_act0-1 ; do
    for Y in AverageRew MaxRew MinRew ; do
        python -m spinup.run plot ${HEAD}/${TITLE}/${TITLE}_s50  --title ${TITLE}  --savefig  --smooth ${SMOOTH}  -x TotalEnvInteracts  -y ${Y}
    done
done
