#include <iostream>
#include <conio.h>
#include "NNetwork.hpp"
#include "Matrix.hpp"
using namespace std;
int main()
{
	NNetwork nn(4, {9, 4, 3, 2});
	while (1)
	{
		while (!_kbhit());
		_getch();
		for (int i = 0; i < 1000; i++) nn.onecycle(Matrix<double>(1, 9, { {0, 0, 0, 0, 1, 1, 1, 1, 0} }), Matrix<double>(1, 2, { {0, 1} }));
		nn.printonscreen();

	}

	return 0;
}