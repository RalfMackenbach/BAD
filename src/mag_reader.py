### This reads various file formats containing magnetic field data
import f90nml
import numpy as np

# Open the file for reading
def read_columns(file_name):
    "Entirely written by ChatGPT."
    # Open the file for reading
    with open(file_name) as f:
        # Find the index of the last line that contains the "/" character
        last_slash_index = None
        for i, line in enumerate(f):
            if "/" in line:
                last_slash_index = i

        # If no "/" character was found, raise an error
        if last_slash_index is None:
            raise ValueError("File does not contain '/' character")

        # Read the remaining lines in the file, starting from the last "/"
        f.seek(0)
        lines = f.readlines()[last_slash_index+1:]

        # Remove any empty lines or comments
        lines = [line for line in lines if line.strip() and not line.strip().startswith('!')]

        # Get the row and column count by counting the number of columns in the first line
        col_count = len(lines[0].split())

        # Create an empty numpy array of the required size
        arr = np.zeros((len(lines), col_count))

        # Loop through the lines in the file
        for i, line in enumerate(lines):
            # Split the line into numbers and convert them to floats
            nums = [float(x) for x in line.split()]
            # Store the numbers in the array
            arr[i, :] = nums

    # Return the array
    return arr



class mag_data:
    """
    Reads magnetic field data.
    """
    def __init__(self, file_name):

        # Let's set all the properties
        file = f90nml.read(file_name)
        params  = file['parameters']
        self.s0     = params['s0']
        self.bref   = params['bref']
        self.my_dpdx= params['my_dpdx']
        self.q0     = params['q0']
        self.shat   = params['shat']
        self.gridpoints = params['gridpoints']
        self.n_pol  = params['n_pol']
        data_arr    = read_columns(file_name)
        self.g11    = data_arr[:,0]
        self.g12    = data_arr[:,1]
        self.g22    = data_arr[:,2]
        self.modB   = data_arr[:,3]
        self.sqrtg  = data_arr[:,4]
        self.L2     = data_arr[:,5]
        self.L1     = data_arr[:,6]
        self.dBdz   = data_arr[:,7]
        self.theta  = np.linspace(-self.n_pol*np.pi,+self.n_pol*np.pi,self.gridpoints,endpoint=False)

    def plot_geometry(self):
        import matplotlib.pyplot as plt
        import matplotlib        as mpl
        font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10}

        mpl.rc('font', **font)
        fig,axs = plt.subplots(2,4 ,tight_layout=True, figsize=(4*3.5, 2*2.5))
        axs[0,0].plot(self.theta/np.pi,self.g11,label='g11')
        axs[0,0].set_title(r'$g_{11}$')
        axs[0,1].plot(self.theta/np.pi,self.g12,label='g12')
        axs[0,1].set_title(r'$g_{12}$')
        axs[0,2].plot(self.theta/np.pi,self.g22,label='g22')
        axs[0,2].set_title(r'$g_{22}$')
        axs[0,3].plot(self.theta/np.pi,self.modB,label='modB')
        axs[0,3].set_title(r'$|B|$')
        axs[1,0].plot(self.theta/np.pi,self.L1,label='L1')
        axs[1,0].set_title(r'$\mathcal{L}_1$')
        axs[1,1].plot(self.theta/np.pi,self.L2,label='L2')
        axs[1,1].set_title(r'$\mathcal{L}_2$')
        axs[1,2].plot(self.theta/np.pi,self.sqrtg,label='sqrtg')
        axs[1,2].set_title(r'$\sqrt{g}$')
        axs[1,3].plot(self.theta/np.pi,self.dBdz,label='dBdz')
        axs[1,3].set_title(r'$\partial_z B$')
        axs[1,0].set_xlabel(r'$\theta/\pi$')
        axs[1,1].set_xlabel(r'$\theta/\pi$')
        axs[1,2].set_xlabel(r'$\theta/\pi$')
        axs[1,3].set_xlabel(r'$\theta/\pi$')
        plt.show()