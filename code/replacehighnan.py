from veusz import embed
import os

# Start an embedded Veusz session
v = veusz.embed.Embedded('myplot')

# Specify the directory containing .npz files
directory = "code\Heisenberg data modified"

# Loop through all .npz files and import them
for filename in os.listdir(directory):
    if filename.endswith(".npz"):
        filepath = os.path.join(directory, filename)
        v.ImportFile(filepath, importer='numpy')
        print(f"Imported: {filename}")

# Save the session or export a plot
v.Export('output_plot.pdf', page=0)
