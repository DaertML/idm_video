# idm_video
Training Inverse Dynamics Models from video in an automated manner

# Introduction
One of the missing pieces to develop further is the auto labeling of data from a little amount of data. The "learning from YouTube" promise that SSL and JEPAs are bringing to the table, still requires a bit of labeled data to match the learned representations to other modalities.

As 69 hours of robots trajectories from the Droid dataset were needed to train the AC VJEPA2 to control a Franka arm... the question arises: can we automate the generation of such dataset by mapping t and t+1 frames to an action, given that we know some minutes of labeled data?

This sounds a lot like synthetic datasets in LLMs, somehow in this case the validation is way easier and the learned semantics are richer.

The first step has been to generate an IDM for the Space Invader game for the Atari; future work has the objective to solve this for robotics, and get similar results to the VJEPA2-AC, but without requiring 69 hours of robot trajectories, and just needing some minutes that will later train an IDM, ad such IDM will create such 60 labeled pairs of frames.

# How to use
Run the gendata.py, which will create vector of data.
Then run the train_idm.py which will train the idm model that will predict the actions (given the action space is known which is a weak assumption).
And run the test_idm.py to see in real time the labeling of the actions taken by the user in the video.
