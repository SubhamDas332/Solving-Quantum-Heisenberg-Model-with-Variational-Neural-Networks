import svgwrite

def draw_ising_lattice_with_alternating_spins():
    # Set up the SVG canvas
    width, height = 600, 200
    dwg = svgwrite.Drawing("ising_lattice_alternating.svg", profile="tiny", size=(width, height))

    # Parameters for the lattice
    num_spins = 8  # Number of spins in the lattice
    spacing = 60  # Horizontal spacing between spins
    y_position = height // 2  # Vertical position for spins
    spin_length = 30  # Length of each spin arrow

    # Loop to create spins with alternating directions
    for i in range(num_spins):
        # Calculate the x position for each spin
        x_position = (i + 1) * spacing

        # Draw the spin arrow (alternating between up and down)
        if i % 2 == 0:
            # Upward spin
            dwg.add(dwg.line(
                start=(x_position, y_position - spin_length),
                end=(x_position, y_position + spin_length),
                stroke="black", stroke_width=3
            ))
            # Upward arrowhead
            dwg.add(dwg.polygon(
                points=[
                    (x_position - 5, y_position - spin_length + 10),
                    (x_position, y_position - spin_length),
                    (x_position + 5, y_position - spin_length + 10)
                ],
                fill="black"
            ))
        else:
            # Downward spin
            dwg.add(dwg.line(
                start=(x_position, y_position + spin_length),
                end=(x_position, y_position - spin_length),
                stroke="black", stroke_width=3
            ))
            # Downward arrowhead
            dwg.add(dwg.polygon(
                points=[
                    (x_position - 5, y_position + spin_length - 10),
                    (x_position, y_position + spin_length),
                    (x_position + 5, y_position + spin_length - 10)
                ],
                fill="black"
            ))

        # Label each spin with an index
        dwg.add(dwg.text(
            str(i), insert=(x_position - 5, y_position + spin_length + 20),
            fill="black", font_size="12px", font_family="Arial"
        ))

    # Save the SVG file
    dwg.save()

# Run the function to create the SVG image
draw_ising_lattice_with_alternating_spins()
