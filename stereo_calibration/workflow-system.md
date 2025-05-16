# Stereo Vision Biomechanical Tracking System

## Workflow Modes & Development Framework

This document outlines a structured system for developing our stereo vision biomechanical tracking system, designed to maintain focus, prevent scattered efforts, and ensure steady progress toward our goals.

## Focus Modes

### Exploratory Mode
```
MODE: Exploratory
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Generate creative solutions, explore possibilities
OUTPUT: Multiple approaches, question assumptions, divergent thinking
PROMPT: "I need to explore options for [specific challenge]. What are different approaches we could consider? I want to think broadly before narrowing down."
```

### Focus Mode
```
MODE: Focus
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Create clear action plan with defined success criteria
OUTPUT: Structured steps, prioritized tasks, decision frameworks
PROMPT: "I need to create a focused plan for [specific task]. What are the essential steps, how should I prioritize them, and how will I know if I've succeeded?"
```

### Problem-Solving Mode
```
MODE: Problem-Solving
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Address specific technical challenges
OUTPUT: Diagnose issues, explain concepts, suggest debugging approaches
PROMPT: "I'm encountering [specific issue] when trying to [task]. I've already tried [previous attempts]. What might be causing this and how can I approach solving it?"
```

### Programming Mode
```
MODE: Programming
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Efficient implementation guidance
OUTPUT: Code examples, algorithm explanations, implementation guidance
PROMPT: "I need to implement [specific feature/function]. Can you help me understand the best approach and provide code guidance for [language/framework]?"
```

### Realignment Mode
```
MODE: Realignment
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Reconnect with project objectives when feeling overwhelmed or off-track
OUTPUT: Strategic perspective, progress assessment, priority redirection
PROMPT: "I'm feeling [overwhelmed/stuck/distracted] with [current situation]. Can we reconnect with our core objectives and determine if I'm still on the right track?"
```

### Capture Planning Mode
```
MODE: Capture Planning
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Plan efficient data collection sessions to maximize value
OUTPUT: Detailed capture protocols, camera settings, session objectives
PROMPT: "I need to plan a capture session for [specific experiment]. What exact settings, conditions, and protocols should I use to ensure I collect all necessary data efficiently?"
```

## Development Workflow Cycle

1. **Define Current Phase Objective** (Realignment Mode)
   - Clarify what specific milestone you're working toward
   - Connect it to longer-term goals
   - Define what "success" looks like

2. **Explore Possible Approaches** (Exploratory Mode)
   - Generate multiple solution paths
   - Consider advantages/disadvantages of each
   - Challenge assumptions

3. **Create Focused Plan** (Focus Mode)
   - Select best approach based on exploration
   - Break down into concrete, sequenced steps
   - Define clear success criteria

4. **Implementation** (Programming Mode)
   - Develop code and algorithms
   - Document as you go
   - Focus on functionality before optimization

5. **Problem Resolution** (Problem-Solving Mode)
   - Address specific issues that arise
   - Debug systematically
   - Understand root causes rather than just symptoms

6. **Capture & Validation** (Capture Planning Mode)
   - Plan precise data collection sessions
   - Ensure all necessary information will be captured
   - Validate against known ground truth when possible

7. **Assess & Iterate** (Return to Realignment Mode)
   - Evaluate progress against success criteria
   - Connect back to broader goals
   - Determine next phase

## Project Progression Path

Our development roadmap moves from fundamental capabilities to increasingly complex human motion tracking:

1. **Calibration Refinement**
   - Integrate larger checkerboard
   - Validate static distance measurements at multiple ranges
   - Achieve <5% calibration error

2. **Motion Tracking Validation**
   - Implement simple pendulum tracking
   - Validate velocity calculations with known trajectories
   - Measure position, velocity, and acceleration accuracy

3. **Basic Pose Estimation**
   - Implement initial pose tracking (MediaPipe or similar)
   - Test with static poses
   - Validate joint angle calculations

4. **Dynamic Movement Analysis**
   - Implement box jump tracking with force plate validation
   - Calculate velocity and force metrics
   - Compare calculated vs. measured forces

5. **Sport-Specific Applications**
   - Implement lead leg block measurement for pitching
   - Add trunk rotation tracking
   - Develop full pitching/batting analysis

## Device Strategy

- **iPhones**: Use for rapid prototyping, workflow development, and initial validation
- **Edgertronics**: Deploy for final validation and precise biomechanical measurements

For each experiment, explicitly decide which technology is appropriate based on:
- Need for precision vs. speed of iteration
- Motion speed of the activity
- Current development phase

## Current Focus

Our immediate focus is on completing calibration validation with the new checkerboard, followed by implementing basic motion tracking using pendulum tests to validate velocity calculations before moving to human subjects.

## Maintaining Context Between Conversations

### Starting a New Conversation
To ensure continuity between conversations, begin new chats with:

```
MODE: Context Restoration
CONTEXT: Stereo Vision Biomechanical Tracking System
FILES: [Include project-update.md and workflow-system.md content]
CURRENT FOCUS: [What you're working on today]
PREVIOUS PROGRESS: [Brief summary of last conversation's outcomes]

After reviewing these documents, let's continue working on [specific task].
```

### Updating Project Status
At the end of significant conversations, request a status update with:

```
MODE: Project Update
CONTEXT: Stereo Vision Biomechanical Tracking System
GOAL: Update project status document with today's progress
PROMPT: "Please update the project status document to reflect what we've accomplished today, the current state of the system, and our next steps."
```

### Continuing Threads with Limited Context
When you need to continue a discussion but are running out of context space:

```
MODE: Thread Continuation
CONTEXT: Stereo Vision Biomechanical Tracking System
PREVIOUS DISCUSSION: [Brief summary of conversation so far]
GOAL: Continue our discussion on [specific topic]
PROMPT: "Let's continue our discussion on [topic]. Here's where we left off: [last few exchanges]"
```

## Project Status Notes

[This section will be updated with each new conversation to maintain context]

- System currently has working intrinsic and extrinsic calibration
- New larger checkerboard being acquired to improve calibration accuracy
- Moving toward controlled motion validation before human tracking
- Planning to use box jumps with force plate validation as first human motion test
- Long-term goal focuses on pitching biomechanics, particularly lead leg block analysis