# Radar-Fire-Dection-with-AI
This is an excerpt from my masters capstone project. We used machine learning techniques to detect smoke plumes within radar-image sequences. For the purposes of this excerpt, I have written code to simulate smoke and non-smoke sequences to avoid the difficultly of storing and downloading large amounts of radar images (as it turns out, due to a severe lack of data, we needed to use simulated smoke sequences in the real project as well).

Running main.py will simulate data and train two models: a simple CNN and an LSTM with CNN inputs. If you're interested to see what the simulations look like, running simulation.py will save a few test sequences in the local directory.
