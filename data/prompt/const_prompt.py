DESCRIBE_PROMPT_BASE="What is in the image?"
DESCRIBE_PROMPT = """What is in this picture? Visual Scene Analysis Prompt
Please analyze the image through these lenses:
1. Objective Observations (Must be strictly based on visible elements):
- Spatial layout: [Cardinal directions/positional relationships]
- Entity inventory: 
  • Primary actors: [Type, quantity, position, motion vectors]
    › (e.g.) "3 pedestrians: NW quadrant, walking 2m/s bearing 120°"
  • Key objects: [Material/texture, size metrics, surface markings]
  • Environmental factors: [Illumination, weather signatures, terrain]
- Dynamic signatures:
  • Velocity gradients: [Acceleration/deceleration patterns]
  • Interaction vectors: [Approaching/diverging trajectories]
  • Biological signals: [Body orientation, limb articulation, gaze vectors]
2. Contextual Speculations (Clearly label as speculation):
- Functional hypotheses:
  • [Object purpose] ← Supported by [shape/texture/contextual alignment]
  • [Device operation] ← Evidence: [user posture/accessory attachments]
- Social dynamics:
  • [Collaborative/conflict indicators] ← Basis: [spatial proximity/gesture patterns]
- Temporal projections:
  • [Next-phase prediction] ← Foundation: [current momentum/obstacle layout]
3. Analysis Hierarchy
Observed → Physical Evidence → Logical Inference → Predictive Extension
Confidence tiers:
✓ Verified (100% image-anchored)
▲ High (≥80% logical consistency)
? Medium (40-79% contextual support)
▽ Low (<40% evidentiary basis)"""

TEST_PROMPT = """You are an advanced AI agent tasked with efficiently and accurately
processing video question and answering tasks.
You will be given a question related to a video, and you are responsibile
for coming up with a set of tactics and plans based on the characteristics
of each question. The questions you will encounter will vary greatly,
ranging from inquires about the overall plot to specific details within
the video.
To effectively handle these tasks, you must first generate a set of tactics
and plans based on the characteristics of each question. You will be given
a question, please analyze the question.
- Determine the type of question (e.g. purpose/goal identification, tools
and materials usage, key action/moment detection, character interaction,
action sequence analysis..etc)
- How should the frames be sampled to solve the question? (e.g. Uniform
sampling with timestep 30. If relevant frame is detected, more uniform
sample with timestep 2.)
context:
{context}
question:
{question}
"""

DEFAULT_PROMPT = """
You are an advanced AI agent tasked with give answer based on the context, give me the reason:
context:
{context}
question:
{question}
"""
