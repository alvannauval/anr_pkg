# If running in Jupyter Notebook, uncomment the line below:
# %matplotlib widget 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Polygon

# --- Geometry Definitions ---
# Full Optical Bounding Volume (Stepped Pocket)
poly_full = np.array([
    [-30, 0], [-20, 0], [-20, -10], [-10, -10], [-10, -30], 
    [10, -30], [10, -10], [20, -10], [20, 0], [30, 0], 
    [30, -40], [-30, -40]
])

# Machining Rule (Bottom Pocket Only, Top deleted)
poly_mach = np.array([
    [-30, -10], [-10, -10], [-10, -30], 
    [10, -30], [10, -10], [30, -10], 
    [30, -40], [-30, -40]
])

# Single Pocket (For easier visualization)
poly_single = np.array([
    [-30, 0], [-10, 0], [-10, -30], 
    [10, -30], [10, 0], [30, 0], 
    [30, -40], [-30, -40]
])

# --- Setup Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.35)
ax.set_xlim(-30, 30)
ax.set_ylim(-35, 10)
ax.set_aspect('equal')
ax.set_title("Optical Bounding Volume vs. Machining Rule", fontsize=14, fontweight='bold')
ax.set_xlabel("Width (mm)")
ax.set_ylabel("Depth (mm)")

# Create the solid metal patch
metal_patch = Polygon(poly_full, closed=True, facecolor='lightgray', edgecolor='black', lw=2)
ax.add_patch(metal_patch)

# Line object for the rays
rays_lines = []

# --- UI Controls ---
ax_angle = plt.axes([0.2, 0.2, 0.5, 0.03])
slider_angle = Slider(ax_angle, 'Zenith Angle (°)', 0.0, 60.0, valinit=30.0, color='orange')

ax_radio = plt.axes([0.1, 0.02, 0.4, 0.15], frameon=False)
radio_mode = RadioButtons(ax_radio, ('Optical Bounding Volume (Correct)', 'Machining Rule (Flawed)', 'Single Pocket (Simple)'))

# --- Optical Physics Engine ---
def get_ray_paths(start_x, angle_deg, mode):
    angle_rad = np.radians(angle_deg)
    path = [(start_x, 10)]
    
    current_x = start_x
    current_y = 10.0
    dx = np.sin(angle_rad)
    dy = -np.cos(angle_rad)
    
    if mode == 'Optical Bounding Volume (Correct)':
        segments = [
            ((-100, 0), (-20, 0)), ((-20, 0), (-20, -10)),
            ((-20, -10), (-10, -10)), ((-10, -10), (-10, -30)),
            ((-10, -30), (10, -30)), ((10, -30), (10, -10)),
            ((10, -10), (20, -10)), ((20, -10), (20, 0)),
            ((20, 0), (100, 0))
        ]
    elif mode == 'Machining Rule (Flawed)':
        segments = [
            ((-100, -10), (-10, -10)), ((-10, -10), (-10, -30)),
            ((-10, -30), (10, -30)), ((10, -30), (10, -10)),
            ((10, -10), (100, -10))
        ]
    else:
        segments = [
            ((-100, 0), (-10, 0)), ((-10, 0), (-10, -30)),
            ((-10, -30), (10, -30)), ((10, -30), (10, 0)),
            ((10, 0), (100, 0))
        ]
        
    for _ in range(10): # Max 10 bounces
        closest_dist = float('inf')
        closest_point = None
        closest_seg = None
        
        for (x1, y1), (x2, y2) in segments:
            den = dx * (y1 - y2) - dy * (x1 - x2)
            if abs(den) < 1e-6:
                continue
                
            t = ((x1 - current_x) * (y1 - y2) - (y1 - current_y) * (x1 - x2)) / den
            u = -((x1 - current_x) * dy - (y1 - current_y) * dx) / den
            
            if t > 1e-5 and 0 <= u <= 1:
                if t < closest_dist:
                    closest_dist = t
                    closest_point = (current_x + t*dx, current_y + t*dy)
                    closest_seg = ((x1, y1), (x2, y2))
                    
        if closest_point:
            path.append(closest_point)
            (x1, y1), (x2, y2) = closest_seg
            if x1 == x2: # Vertical wall bounce
                dx = -dx
                current_x, current_y = closest_point
            else: # Floor hit
                break
        else:
            path.append((current_x + 100*dx, current_y + 100*dy))
            break
            
    return path

# --- Update Logic ---
def update(val):
    angle = slider_angle.val
    mode = radio_mode.value_selected
    
    # Update solid geometry
    if mode == 'Optical Bounding Volume (Correct)':
        metal_patch.set_xy(poly_full)
    elif mode == 'Machining Rule (Flawed)':
        metal_patch.set_xy(poly_mach)
    else:
        metal_patch.set_xy(poly_single)
        
    # Clear old rays
    while len(ax.lines) > 0:
        ax.lines[0].remove()
        
    bottom_hits = 0
    total_rays = 0
        
    # Cast 40 parallel rays (simulating active stereo projector)
    for start_x in np.linspace(-40, 20, 40):
        total_rays += 1
        path = get_ray_paths(start_x, angle, mode)
        end_x, end_y = path[-1]
        
        # Determine ray color based on whether it bounced and where it lands
        if len(path) > 2 and end_y <= -29 and path[1][1] > -10:  # Bounce off 1st pocket wall reaching pocket 2
            color = 'green'
            bottom_hits += 1
        else:
            color = 'red'
            
        # Draw the ray and the impact point
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, alpha=0.5, lw=1.5)
        ax.plot(end_x, end_y, marker='o', color=color, markersize=4)
        if len(path) > 2:
            ax.plot(xs[1:-1], ys[1:-1], marker='x', color='orange', markersize=4, linestyle='None')

    # Dynamic text update to prove the thesis point
    fig.suptitle(f"Points from secondary bounces: {bottom_hits} / {total_rays}", 
                 fontsize=12, color='green' if bottom_hits > 0 else 'red', y=0.95)
                 
    fig.canvas.draw_idle()

# Link UI to update function
slider_angle.on_changed(update)
radio_mode.on_clicked(update)

# Initialize plot
update(0)
plt.show()