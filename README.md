# Radar-Fire-Dection-with-AI
This is an excerpt from a my masters capstone project in which we used machine learning techniques to identify radar image sequences which contain smoke plumes. For the purposes of this excerpt, I have written code to simulate smoke and non-smoke sequences to avoid the difficultly of storing and downloading large amounts of radar images (as it turns out, we needed to use simulated smoke sequences in the real project as well due to a severe lack of data).

Running main.py will simulate data and train two models: a simple CNN and an LSTM with CNN inputs. If you're interested to see what the simulations look like, running simulation.py will save a few test sequences in the local directory.
