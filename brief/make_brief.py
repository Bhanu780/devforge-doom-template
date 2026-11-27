"""Generate the two-page submission brief using Python and Ghostscript."""
import textwrap,subprocess
from pathlib import Path
lines=['%!PS-Adobe-3.0','<< /PageSize [595 842] >> setpagedevice']
def label(x,y,s,size=12,bold=False,color='0.10 0.17 0.22'):
 s=s.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
 lines.append(f'{color} setrgbcolor /Helvetica{"-Bold" if bold else ""} findfont {size} scalefont setfont {x} {y} moveto ({s}) show')
def start(page,title,sub):
 lines.append('0.04 0.18 0.22 setrgbcolor 0 707 595 135 rectfill')
 label(44,802,'YUGAANTAR DEVFORGE  /  TRACK 1',11,True,'0.48 0.86 0.79')
 label(44,754,title,29,True,'1 1 1')
 label(44,728,sub,11,False,'0.86 0.95 0.94')
 label(44,35,'MapMind   /   Can It Play DOOM?',10)
 label(485,35,f'{page}  /  2',10)
def section(y,h,p):
 label(44,y,h,16,True,'0.02 0.36 0.34')
 y-=23
 for row in textwrap.wrap(p,87):
  label(44,y,row,11.7)
  y-=17
 return y-19
start(1,'MapMind','An observation-driven agent for combat, exploration and multi-room progression')
y=section(668,'Remembering a door after leaving the room','A locked door and its key rarely appear in the same view. MapMind records doors, keys and explored space in a map built from its own observations. A recurrent policy reads that map alongside the current scene to choose its next action. The policy can revisit a previously seen location even after the original observation has disappeared.')
y=section(y,'From a game observation to a button press','Every four engine ticks, the official ViZDoom wrapper supplies 64 rays, visible entities, player vitals, RGB and depth. A circular convolution encodes the rays. Masked attention combines the visible entities, while convolutional encoders process the visual input and the memory map. A 512-unit GRU integrates the fused features over time. The actor samples nine button outputs. A value head estimates future return during training.')
y=section(y,'Spatial memory and wall contact','The agent estimates heading from its turn inputs and displacement from player-relative velocity. Ray comparisons help detect pushing against a wall despite reported momentum. It writes free space, walls, doors, locks, items, keys, enemies and visits into eight map channels. The policy reads a 32-by-32 local crop rotated into its current viewing direction. Position errors can accumulate over long routes.')
y=section(y,'Inference contract','DoomAgent loads PyTorch weights once, clears its map and recurrent state on reset, and chooses actions from the official observation. Observation-based masks suppress unavailable or ineffective actions such as distant punches. Demonstration routes and privileged engine state are confined to training.')
assert y>45
lines.append('showpage')
start(2,'Teaching the final approach','Training pipeline, controlled comparisons and reproducibility')
y=section(670,'Training for the exit','Recurrent PPO trains on Freedoom, combat arenas and procedural keycard layouts. Rewards credit exits, keys, newly opened doors, exploration and combat outcomes. Repeated visits earn diminishing curiosity bonuses. To practice finishing, we fine-tuned a saved policy on 48 keycard demonstrations, 24 successful arena exit demonstrations and eight combat episodes. One imitation epoch used 8,003 decisions. We then ran 4,096 PPO decisions with a mixture of exit practice and full maps.')
label(44,y,'Held-out exit performance',16,True,'0.02 0.36 0.34')
y-=32
for i,row in enumerate([('Checkpoint','2,100 ticks','4,200 ticks','Two-lock*'),('Starting policy','0 / 18','6 / 18','0 / 6'),('Exit-trained policy','6 / 18','9 / 18','3 / 6')]):
 lines.append(f'0.92 0.96 0.95 setrgbcolor 40 {y-7} 510 24 rectfill')
 for x,v in zip([47,240,344,451],row):label(x,y,v,11,i==0)
 y-=29
y=section(y-7,'What the measurements cover','Nine held-out generated layouts were run with two seeds each. Counts show episodes reaching the exit. *Two-lock counts are a subset at 4,200 ticks. The policy survived all episodes at both limits. On 12 additional layouts from a fresh generator seed, it cleared 3 and survived all 12 at 2,100 ticks. Both Freedoom MAP01 runs timed out. On reserved MAP04, one run timed out and one ended in death. These are local results.')
y=section(y,'Reproducing the submission','Clone the public repository with Git LFS, run git lfs pull, then run python validate.py --device cpu in the project environment. The entry point is submission/agent.py and the checkpoint is submission/weights/policy.pt. Training code is in training/imitate_keycard.py and training/train.py.')
label(44,y,'github.com/Bhanu780/devforge-doom-template',10,True)
assert y>50
lines.append('showpage')
subprocess.run(['ps2pdf','-',str(Path(__file__).with_name('Project_Brief.pdf'))],input='\n'.join(lines).encode(),check=True)
