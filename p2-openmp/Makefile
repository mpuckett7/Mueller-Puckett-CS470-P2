default: gauss par_gauss par_gauss_serial

gauss: gauss.c timer.h
	gcc -g -O3 --std=c99 -Wall -o gauss gauss.c

par_gauss: par_gauss.c timer.h
	gcc -g -O3 --std=c99 -fopenmp -Wall -o par_gauss par_gauss.c

par_gauss_serial: par_gauss.c timer.h
	gcc -g -O3 --std=c99 -Wno-unknown-pragmas -Wall -o par_gauss_serial par_gauss.c

clean:
	rm -f gauss par_gauss par_gauss_serial

