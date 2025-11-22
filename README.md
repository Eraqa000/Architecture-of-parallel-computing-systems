Если пользуешься linux:
  Для mpi:
  
    можно компилировать с: mpcc "file_name".c -o file.exe
    запускать: mpiexec -n 4 file.exe
    
  Для архитектуры CUDA (можно запускать если видеокарта от NVIDIA):
  
    можно компилировать с: mvcc "file_name".cu -o file
    Запускать: ./file

Если пользуетесь с windows:
  Для mpi:
  
    можно компилировать с: gcc "file_name".c -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -o "file".exe
    запускать: mpiexec -n 4 "file".exe
  
