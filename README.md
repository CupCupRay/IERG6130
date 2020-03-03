# IERG6130 Project
## Learning to Evade Jamming Attacks Using Reinforcement Learning
***
### Environment

Create the environment

```python
env = Env(MAX_channel, Total_packets)
# MAX_channel: Max number of channels
# Total_packets: Number of packets need to be send
```

Change the attacker's strategy

```python
def mode: # Attcker's strategy
    0: # No attack
    1: # Randomly choose only ONE channel to attack
    2: # Randomly choose only HALF channels to attack
```

Reset the environment

```python
env.reset(mode)
```

Interaction with environment

```python
state, reward, done, info = env.step(action)
```

Output the time cost for sending all the packets

```python
print(env.time)
```

