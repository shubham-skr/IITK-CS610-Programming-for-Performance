#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)


void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main()
{
  int i, j;

  i = 0;
  FILE *fp = fopen("./disp.txt", "r");
  if (fp == NULL)
  {
    printf("Error: could not open file\n");
    return 1;
  }

  for (int i = 0; i < 120 && fscanf(fp, "%lf", &a[i]) == 1; ++i)
    ;
  fclose(fp);

  // read grid file
  j = 0;
  FILE *fpq = fopen("./grid.txt", "r");
  if (fpq == NULL)
  {
    printf("Error: could not open file\n");
    return 1;
  }

  for (int j = 0; j < 30 && fscanf(fpq, "%lf", &b[j]) == 1; ++j)
    ;
  fclose(fpq);

  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                 b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                 b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                 a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
                 a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31],
                 a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                 a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55],
                 a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63], a[64], a[65], a[66], a[67],
                 a[68], a[69], a[70], a[71], a[72], a[73], a[74], a[75], a[76], a[77], a[78], a[79],
                 a[80], a[81], a[82], a[83], a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91],
                 a[92], a[93], a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102],
                 a[103], a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                 a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}


void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk)
{

  register double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;

  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  long pnts = 0;

  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;

 
  e1 = kk * ey1;
  e2 = kk * ey2;
  e3 = kk * ey3;
  e4 = kk * ey4;
  e5 = kk * ey5;
  e6 = kk * ey6;
  e7 = kk * ey7;
  e8 = kk * ey8;
  e9 = kk * ey9;
  e10 = kk * ey10;


  register int s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
  s1 = floor((dd2 - dd1) / dd3);
  s2 = floor((dd5 - dd4) / dd6);
  s3 = floor((dd8 - dd7) / dd9);
  s4 = floor((dd11 - dd10) / dd12);
  s5 = floor((dd14 - dd13) / dd15);
  s6 = floor((dd17 - dd16) / dd18);
  s7 = floor((dd20 - dd19) / dd21);
  s8 = floor((dd23 - dd22) / dd24);
  s9 = floor((dd26 - dd25) / dd27);
  s10 = floor((dd29 - dd28) / dd30);

  
   FILE *fptr = fopen("./results_v1.txt", "w");
    if (fptr == NULL)
    {
        printf("Error in creating file !");
        exit(1);
    }
  
	
  x1 = dd1;
  for (int r1 = 0; r1 < s1; ++r1, x1 += dd3)
  {
    double p1_1 = c11 * x1 - d1;
    double p2_1 = c21 * x1 - d2;
    double p3_1 = c31 * x1 - d3;
    double p4_1 = c41 * x1 - d4;
    double p5_1 = c51 * x1 - d5;
    double p6_1 = c61 * x1 - d6;
    double p7_1 = c71 * x1 - d7;
    double p8_1 = c81 * x1 - d8;
    double p9_1 = c91 * x1 - d9;
    double p10_1 = c101 * x1 - d10;

    x2 = dd4;
    for (int r2 = 0; r2 < s2; ++r2, x2 += dd6)
    {
      double p1_2 = p1_1 + c12 * x2;
      double p2_2 = p2_1 + c22 * x2;
      double p3_2 = p3_1 + c32 * x2;
      double p4_2 = p4_1 + c42 * x2;
      double p5_2 = p5_1 + c52 * x2;
      double p6_2 = p6_1 + c62 * x2;
      double p7_2 = p7_1 + c72 * x2;
      double p8_2 = p8_1 + c82 * x2;
      double p9_2 = p9_1 + c92 * x2;
      double p10_2 = p10_1 + c102 * x2;

      x3 = dd7;
      for (int r3 = 0; r3 < s3; ++r3, x3 += dd9)
      {
        double p1_3 = p1_2 + c13 * x3;
        double p2_3 = p2_2 + c23 * x3;
        double p3_3 = p3_2 + c33 * x3;
        double p4_3 = p4_2 + c43 * x3;
        double p5_3 = p5_2 + c53 * x3;
        double p6_3 = p6_2 + c63 * x3;
        double p7_3 = p7_2 + c73 * x3;
        double p8_3 = p8_2 + c83 * x3;
        double p9_3 = p9_2 + c93 * x3;
        double p10_3 = p10_2 + c103 * x3;

        x4 = dd10;
        for (int r4 = 0; r4 < s4; ++r4, x4 += dd12)
        {
          double p1_4 = p1_3 + c14 * x4;
          double p2_4 = p2_3 + c24 * x4;
          double p3_4 = p3_3 + c34 * x4;
          double p4_4 = p4_3 + c44 * x4;
          double p5_4 = p5_3 + c54 * x4;
          double p6_4 = p6_3 + c64 * x4;
          double p7_4 = p7_3 + c74 * x4;
          double p8_4 = p8_3 + c84 * x4;
          double p9_4 = p9_3 + c94 * x4;
          double p10_4 = p10_3 + c104 * x4;

          x5 = dd13;
          for (int r5 = 0; r5 < s5; ++r5, x5 += dd15)
          {
            double p1_5 = p1_4 + c15 * x5;
            double p2_5 = p2_4 + c25 * x5;
            double p3_5 = p3_4 + c35 * x5;
            double p4_5 = p4_4 + c45 * x5;
            double p5_5 = p5_4 + c55 * x5;
            double p6_5 = p6_4 + c65 * x5;
            double p7_5 = p7_4 + c75 * x5;
            double p8_5 = p8_4 + c85 * x5;
            double p9_5 = p9_4 + c95 * x5;
            double p10_5 = p10_4 + c105 * x5;

            x6 = dd16;
            for (int r6 = 0; r6 < s6; ++r6, x6 += dd18)
            {
              double p1_6 = p1_5 + c16 * x6;
              double p2_6 = p2_5 + c26 * x6;
              double p3_6 = p3_5 + c36 * x6;
              double p4_6 = p4_5 + c46 * x6;
              double p5_6 = p5_5 + c56 * x6;
              double p6_6 = p6_5 + c66 * x6;
              double p7_6 = p7_5 + c76 * x6;
              double p8_6 = p8_5 + c86 * x6;
              double p9_6 = p9_5 + c96 * x6;
              double p10_6 = p10_5 + c106 * x6;

              x7 = dd19;
              for (int r7 = 0; r7 < s7; ++r7, x7 += dd21)
              {
                double p1_7 = p1_6 + c17 * x7;
                double p2_7 = p2_6 + c27 * x7;
                double p3_7 = p3_6 + c37 * x7;
                double p4_7 = p4_6 + c47 * x7;
                double p5_7 = p5_6 + c57 * x7;
                double p6_7 = p6_6 + c67 * x7;
                double p7_7 = p7_6 + c77 * x7;
                double p8_7 = p8_6 + c87 * x7;
                double p9_7 = p9_6 + c97 * x7;
                double p10_7 = p10_6 + c107 * x7;

                x8 = dd22;
                for (int r8 = 0; r8 < s8; ++r8, x8 += dd24)
                {
                  double p1_8 = p1_7 + c18 * x8;
                  double p2_8 = p2_7 + c28 * x8;
                  double p3_8 = p3_7 + c38 * x8;
                  double p4_8 = p4_7 + c48 * x8;
                  double p5_8 = p5_7 + c58 * x8;
                  double p6_8 = p6_7 + c68 * x8;
                  double p7_8 = p7_7 + c78 * x8;
                  double p8_8 = p8_7 + c88 * x8;
                  double p9_8 = p9_7 + c98 * x8;
                  double p10_8 = p10_7 + c108 * x8;

                  x9 = dd25;
                  for (int r9 = 0; r9 < s9; ++r9, x9 += dd27)
                  {
                    double p1_9 = p1_8 + c19 * x9;
                    double p2_9 = p2_8 + c29 * x9;
                    double p3_9 = p3_8 + c39 * x9;
                    double p4_9 = p4_8 + c49 * x9;
                    double p5_9 = p5_8 + c59 * x9;
                    double p6_9 = p6_8 + c69 * x9;
                    double p7_9 = p7_8 + c79 * x9;
                    double p8_9 = p8_8 + c89 * x9;
                    double p9_9 = p9_8 + c99 * x9;
                    double p10_9 = p10_8 + c109 * x9;

                    x10 = dd28;
                    for (int r10 = 0; r10 < s10; ++r10, x10 += dd30)
                    {

                      double q1 = fabs(p1_9 + c110 * x10);
                      if (q1 > e1)
                       continue;

                      double q2 = fabs(p2_9 + c210 * x10);
                      if (q2 > e2)
                        continue;

                      double q3 = fabs(p3_9 + c310 * x10);
                      if (q3 > e3)
                        continue;

                      double q4 = fabs(p4_9 + c410 * x10);
                      if (q4 > e4)
                        continue;

                      double q5 = fabs(p5_9 + c510 * x10);
                      if (q5 > e5)
                        continue;

                      double q6 = fabs(p6_9 + c610 * x10);
                      if (q6 > e6)
                        continue;

                      double q7 = fabs(p7_9 + c710 * x10);
                      if (q7 > e7)
                        continue;

                      double q8 = fabs(p8_9 + c810 * x10);
                      if (q8 > e8)
                        continue;

                      double q9 = fabs(p9_9 + c910 * x10);
                      if (q9 > e9)
                        continue;

                      double q10 = fabs(p10_9 + c1010 * x10);
                      if (q10 > e10)
                        continue;
			
		      	
                      pnts++;

                 
                     fprintf(fptr, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  fclose(fptr);
 
  printf("result pnts: %ld\n", pnts);
}
