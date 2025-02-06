SO-NMF Python Implementation

This repository contains the Python implementation of SO-NMF, as described in the paper:
“Synergistic Functional Spectrum Analysis: A Framework for Exploring the Multifunctional Interplay Among Multimodal Nonverbal Behaviours in Conversations”
Published in IEEE Transactions on Affective Computing (doi: 10.1109/TAFFC.2024.3491097).
This code has been verified with Python 3.8.16.

1. Overview
    SO-NMF (Semi Orthogonal Non-negative Matrix Factorization) is the core computational module employed in the framework of synergistic functional spectrum analysis (sFSA), dedicated to analyzing the communicative functions of multimodal nonverbal behaviors that appear in human conversations.

2. Preliminary preparation
    Dependency: onmf.py
        This project uses the onmf.py module from the torch-onmf repository to implement Orthogonal Non-negative Matrix Factorization (O-NMF).
    Steps:
        1. Download the onmf.py file from the following link: https://github.com/signofyouth/torch-onmf/blob/main/onmf.py
        2. Import onmf.py into your project directory.

3. Example of use
    Sample data: 
        ・File: sample_data_for_SONMF.txt
        ・Description: The file contains 24 columns of waveform data with a length of 19500 frames, each representing different amplitudes and frequencies. This waveform data is just dummy data. As we mentioned in our paper, the actual data used in the paper is not publicly available. We plan to create another dataset and make it publicly available in 2026 (limited to academic research in universities or public research institutes. The detailed licensing is TBA). 
    Output: 
        Running SO-NMF on the sample data will generate the following output files:
            WIN.npy: The orthonormal basis matrix (the initial basis matrix for the NMF).
            W.npy: The basis matrix resulting from SO-NMF.
            H.npy: The coefficient matrix resulting from SO-NMF.
            WH.npy: The product of the basis and coefficient matrices.
    Note:
        The default number of bases in SO-NMF.py is set to 6, as per the paper. However, this parameter can be adjusted based on user requirements.

5. Citation
    If you use any of the resources provided on this page in any of your publications, we ask you to cite the following work.
    M. Imamura, A. Tashiro, S. Kumano and K. Otsuka, "Synergistic Functional Spectrum Analysis: A Framework for Exploring the Multifunctional Interplay Among Multimodal Nonverbal Behaviours in Conversations," IEEE Transactions on Affective Computing, 2024, doi: 10.1109/TAFFC.2024.3491097.

4. Article Link
    For more information, please refer to the original paper: https://ieeexplore.ieee.org/document/10742505