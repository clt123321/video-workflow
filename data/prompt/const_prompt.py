#################################### VLM 提示词 ##############################################VLM_DESCRIBE_PROMPT_BASE = "What is in the image?"
VLM_IMAGE_CAPTION_PROMPT_BASE = "Give me a caption about this image"
VLM_DESCRIBE_PROMPT = """What is in this picture? Visual Scene Analysis Prompt
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

#################################### LLM 提示词 ##############################################
# 分析问题的类型，选择采样方式
# from avua, Policy Generation Prompt
LLM_POLICY_GENERATION_PROMPT = """You are an advanced AI agent tasked with efficiently and accurately
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

LLM_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant designed to output JSON."
LLM_SAMPLE_FRAME_WITH_CONTEXT_PROMPT = """
Given a video with {num_frames} frames (decoded at 1 fps) and the following contextual information:
{context}

Additional video understanding markers:
- #C: Action performed by camera wearer
- #O: Action performed by others

Based on the contextual understanding and temporal relationships between events, please select the most relevant frame(s) to answer this question:
{question}

Guidelines:
1. Analyze the temporal progression of events
2. Identify key moments that could visually answer the question
3. Consider both immediate context and long-term dependencies
4. Determine required visual evidence type (object, action, spatial relation, etc.)

# Return a JSON format {answer_format}
Example response for reference:
```json
{{
    "frame_id": 62,
    "vlm_prompt": "Focus on the beaker in the colleague's hands. Describe the exact color and opacity of the liquid mixture at the moment of initial combining."
}}
``` 
"""

LLM_GIVE_ANSER_BY_CAPTION_PROMPT = """
Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
{caption}
#C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
#O to denote that the sentence is an action done by someone other than the camera wearer.
Please answer the following question: 
``` 
{question}
``` 
Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
"""

LLM_SELF_EVAL_PROMPT = """
Please assess the confidence level in the decision-making process.
The provided information is as as follows,
{previous_information}
The decision making process is as follows,
{answer}
Criteria for Evaluation:
Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
Partial Information (Confidence Level: 2): If information partially supports an informed guess.
Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
Assessment Focus:
Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
Please generate the confidence with JSON format {confidence_format}
"""

LLM_GET_ANSWER_SYSTEM_PROMPT = """
You are an advanced AI agent tasked with give answer based on the context, give me the reason:
context:
{context}
question:
{question}
"""
