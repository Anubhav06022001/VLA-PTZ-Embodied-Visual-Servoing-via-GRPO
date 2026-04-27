import matplotlib.pyplot as plt

# ------------------ Data ------------------
epochs = [
    50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
    550, 600, 650, 700, 750, 800, 850, 900, 950, 1000
]

rewards = [
    280, 304, 640, 484, 436, 496, 748, 556, 640, 724,
    520, 772, 604, 772, 556, 712, 724, 832, 796, 748
]

# ------------------ Plot ------------------
plt.figure(figsize=(10, 6))

plt.plot(epochs, rewards, marker='o')

# Labels and title
plt.title("Training Performance: Total Reward vs Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)

# Grid for readability
plt.grid(True)

# Tight layout for clean spacing
plt.tight_layout()

# Save figure (important for submission)
plt.savefig("training_curve.png", dpi=300)

# Show plot
plt.show()