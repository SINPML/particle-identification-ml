# particle-identification-ml
ML models for improving particle identification in experiments conducted using CLAS12 detector  
Abstract:

**Machine Learning for Particle Identification in CLAS12: Random Forest Approach for Exclusive Reactions**

Filatova Veronika Anatolyevna 
Student of Lomonosov Moscow State University, Faculty of Physics, Moscow, Russia  
E-mail: filatova.va22@physics.msu.ru

Every year, methods of artificial intelligence and machine learning in particular become more and more powerful tools for studying various fields of science. Elementary particle physics is not an exception. Machine learning algorithms can already be used to build data generators, suppress noise and background processes in data, determine particle tracks, and identify particle types. Researchers working with CLAS12 detector at Jefferson Lab deal with large amounts of data that allow efficient construction of machine learning models. The variety of detector subsystems (DC, FTOF, EC, etc.) provides information about particle momenta, energies, and flight times, which subsequently allows the extraction of reaction cross sections and identification of final-state particles.

In this paper, we study the methods of machine learning in the task of particle identification (PID) for exclusive reactions detected by the CLAS12 spectrometer. When considering the reactions:

(1) `e- + p -> e- + n + pi+`  
(2) `e- + p -> e- + p + pi+ + pi-`

data from CLAS12 experimental runs were used to train a classification model capable of distinguishing pions, protons, and other particles in the final state. The algorithm used in this work is a **Random Forest classifier**, which was trained on kinematic variables such as momentum, energy loss, time-of-flight, and detector hit information. In addition to standard classification metrics (accuracy, precision, recall, F1-score), this paper also presents physically based comparisons of PID performance as a function of momentum and scattering angle. Confusion matrices and efficiency curves are provided to validate the model's capability.

Based on this algorithm, it is possible to significantly improve particle identification purity and efficiency compared to traditional cuts-based approaches, especially in kinematical regions where particle types overlap. This work demonstrates the potential of ensemble learning methods for enhancing the physics reach of the CLAS12 experiment, particularly for exclusive channel reconstruction.
