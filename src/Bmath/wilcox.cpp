/*
  Copyright (C) 2005 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

/*
  Mathlib : A C Library of Special Functions
  Copyright (C) 1999-2000  The R Development Core Team

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or (at
  your option) any later version.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.

  SYNOPSIS

    #include <Bmath.hpp>
    double dwilcox(double x, double m, double n, int give_log)
    double pwilcox(double x, double m, double n, int lower_tail, int log_p)
    double qwilcox(double x, double m, double n, int lower_tail, int log_p);
    double rwilcox(double m, double n)

  DESCRIPTION

    dwilcox	The density of the Wilcoxon distribution.
    pwilcox	The distribution function of the Wilcoxon distribution.
    qwilcox	The quantile function of the Wilcoxon distribution.
    rwilcox	Random variates from the Wilcoxon distribution.

 */

#include "nmath.hpp"
#include "dpq.hpp"
#include <cstdlib>
namespace Rmath{

#ifndef MATHLIB_STANDALONE
#ifdef Macintosh
extern void isintrpt();
#endif
#ifdef Win32
extern void R_ProcessEvents();
#endif
#endif

static double ***w;
static int allocated_m, allocated_n;

static void
w_free(int m, int n)
{
    int i, j;

    if (m > n) {
	i = n; n = m; m = i;
    }
    m = std::max(m, WILCOX_MAX);
    n = std::max(n, WILCOX_MAX);

    for (i = m; i >= 0; i--) {
	for (j = n; j >= 0; j--) {
	    if (w[i][j] != 0)
		free((void *) w[i][j]);
	}
	free((void *) w[i]);
    }
    free((void *) w);
    w = 0; allocated_m = allocated_n = 0;
}

static void
w_init_maybe(int m, int n)
{
    int i;

    if (w && (m > WILCOX_MAX || n > WILCOX_MAX))
	w_free(WILCOX_MAX, WILCOX_MAX);

    if (!w) {
	allocated_m = m; allocated_n = n;
	if (m > n) {
	    i = n; n = m; m = i;
	}
	m = std::max(m, WILCOX_MAX);
	n = std::max(n, WILCOX_MAX);
	w = (double ***) calloc(m + 1, sizeof(double **));
	if (!w)
	    mathlib_error("wilcox allocation error", 1);
	for (i = 0; i <= m; i++) {
	    w[i] = (double **) calloc(n + 1, sizeof(double *));
	    if (!w[i])
		mathlib_error("wilcox allocation error ", 2);
	}
    }
}

inline void W_INIT_MAYBE(double M, double N){
  int m =static_cast<int>(M);
  int n =static_cast<int>(N);
  w_init_maybe(m,n);
}

static void
w_free_maybe(int m, int n)
{
    if (m > WILCOX_MAX || n > WILCOX_MAX)
	w_free(m, n);
}

static double
cwilcox(int k, int m, int n)
{
    int c, u, i, j, l;

#ifndef MATHLIB_STANDALONE
    /* check for a user interrupt */
#ifdef Macintosh
    isintrpt();
#endif
#ifdef Win32
    R_ProcessEvents();
#endif
#endif

    u = m * n;
    c = (int)(u / 2);

    if ((k < 0) || (k > u))
	return(0);
    if (k > c)
	k = u - k;
    if (m < n) {
	i = m; j = n;
    } else {
	i = n; j = m;
    }

    if (w[i][j] == 0) {
	w[i][j] = (double *) calloc(c + 1, sizeof(double));
	if (!w[i][j])
		mathlib_error("wilcox allocation error ", 3);
	for (l = 0; l <= c; l++)
	    w[i][j][l] = -1;
    }
    if (w[i][j][k] < 0) {
	if ((i == 0) || (j == 0))
	    w[i][j][k] = (k == 0);
	else
	    w[i][j][k] = cwilcox(k - n, m - 1, n)
		+ cwilcox(k, m, n - 1);

    }
    return(w[i][j][k]);
}

inline double CWILCOX(double K, double M, double N){
  int k = static_cast<int>(K);
  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  return cwilcox(k,m,n);
}

double dwilcox(double x, double m, double n, int give_log)
{
    double d;

#ifdef IEEE_754
    /* NaNs propagated correctly */
    if (ISNAN(x) || ISNAN(m) || ISNAN(n))
	return(x + m + n);
#endif
    m = FLOOR(m + 0.5);
    n = FLOOR(n + 0.5);
    if (m <= 0 || n <= 0)
	ML_ERR_return_NAN;

    if (fabs(x - FLOOR(x + 0.5)) > 1e-7)
	return(R_D__0);
    x = FLOOR(x + 0.5);
    if ((x < 0) || (x > m * n))
	return(R_D__0);

    W_INIT_MAYBE(m, n);
    d = give_log ?
	log(CWILCOX(x, m, n)) - lchoose(m + n, n) :
	    CWILCOX(x, m, n)  /	 choose(m + n, n);

    return(d);
}

double pwilcox(double x, double m, double n, int lower_tail, int log_p)
{
    int i;
    double c, p;

#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(m) || ISNAN(n))
	return(x + m + n);
#endif
    if (!R_FINITE(m) || !R_FINITE(n))
	ML_ERR_return_NAN;
    m = FLOOR(m + 0.5);
    n = FLOOR(n + 0.5);
    if (m <= 0 || n <= 0)
	ML_ERR_return_NAN;

    x = FLOOR(x + 1e-7);

    if (x < 0.0)
	return(R_DT_0);
    if (x >= m * n)
	return(R_DT_1);

    W_INIT_MAYBE(m, n);
    c = choose(m + n, n);
    p = 0;
    if (x <= (m * n / 2)) {
	for (i = 0; i <= x; i++)
	    p += CWILCOX(i, m, n) / c;
    }
    else {
	x = m * n - x;
	for (i = 0; i < x; i++)
	    p += CWILCOX(i, m, n) / c;
	lower_tail = !lower_tail; /* p = 1 - p; */
    }

    return(R_DT_val(p));
} /* pwilcox */

double qwilcox(double x, double m, double n, int lower_tail, int log_p)
{
    double c, p, q;

#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(m) || ISNAN(n))
	return(x + m + n);
#endif
    if(!R_FINITE(x) || !R_FINITE(m) || !R_FINITE(n))
	ML_ERR_return_NAN;
    R_Q_P01_check(x);

    m = FLOOR(m + 0.5);
    n = FLOOR(n + 0.5);
    if (m <= 0 || n <= 0)
	ML_ERR_return_NAN;

    if (x == R_DT_0)
	return(0);
    if (x == R_DT_1)
	return(m * n);

    if(log_p || !lower_tail)
	x = R_DT_qIv(x); /* lower_tail,non-log "p" */

    W_INIT_MAYBE(m, n);
    c = choose(m + n, n);
    p = 0;
    q = 0;
    if (x <= 0.5) {
	x = x - 10 * numeric_limits<double>::epsilon();
	for (;;) {
	    p += CWILCOX(q, m, n) / c;
	    if (p >= x)
		break;
	    q++;
	}
    }
    else {
	x = 1 - x + 10 * numeric_limits<double>::epsilon();
	for (;;) {
	    p += CWILCOX(q, m, n) / c;
	    if (p > x) {
		q = m * n - q;
		break;
	    }
	    q++;
	}
    }

    return(q);
}

double rwilcox(double m, double n)
{
    int i, j, k, *x;
    double r;

#ifdef IEEE_754
    /* NaNs propagated correctly */
    if (ISNAN(m) || ISNAN(n))
	return(m + n);
#endif
    m = FLOOR(m + 0.5);
    n = FLOOR(n + 0.5);
    if ((m < 0) || (n < 0))
	ML_ERR_return_NAN;

    if ((m == 0) || (n == 0))
	return(0);

    r = 0.0;
    k = (int) (m + n);
    x = (int *) calloc(k, sizeof(int));
    if (!x)
	mathlib_error("wilcox allocation error ", 4);
    for (i = 0; i < k; i++)
	x[i] = i;
    for (i = 0; i < n; i++) {
        j = FLOOR(k * unif_rand(BOOM::GlobalRng::rng));
	r += x[j];
	x[j] = x[--k];
    }
    free(x);
    return(r - n * (n - 1) / 2);
}

void wilcox_free()
{
    w_free_maybe(allocated_m, allocated_n);
}


}
