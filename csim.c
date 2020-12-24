#include "cachelab.h"
#include<stdlib.h>
#include<unistd.h>
#include<stdio.h>
#include<limits.h>
#include<getopt.h>
#include<string.h> 
int help_mode, verbose_mode, s, E, b, S,number_hits, number_miss, number_eviction;
char filename[1000];
char buffer[1000];
typedef struct 
{
    int valid_bit, tag, stamp;
}cache_line;
cache_line **cache = NULL;

void update(unsigned int address)
{
    int max_stamp = INT_MIN, max_stamp_id = -1;
    int  t_address, s_address;
    s_address = (address >> b) & ((-1U) >> (32 - s));
    t_address = address >> (s + b);
    for(int i = 0; i < E; i++)
        if(cache[s_address][i].tag == t_address)
        {
            cache[s_address][i].stamp = 0;
            number_hits++;
            return ;
        }
    for(int i = 0; i < E; i++)
        if(cache[s_address][i].valid_bit == 0)
        {
            cache[s_address][i].valid_bit = 1;
            cache[s_address][i].tag = t_address;
            cache[s_address][i].stamp = 0;
            number_miss++;
            return;
        }
    number_eviction++;
    number_miss++;
    for(int i = 0; i < E; i++)
        if(cache[s_address][i].stamp > max_stamp)
        {
            max_stamp = cache[s_address][i].stamp;
            max_stamp_id = i;
        }
    cache[s_address][max_stamp_id].tag = t_address;
    cache[s_address][max_stamp_id].stamp = 0;
    return;
}

void update_time(void)
{
    for(int i = 0; i < S; i++)
        for(int j = 0; j < E; j++)
            if(cache[i][j].valid_bit == 1)
                cache[i][j].stamp++;
}

int main(int argc,char* argv[])
{
    int opt, temp;
    char type;
    unsigned int address;
    number_hits = number_miss = number_eviction = 0;
    while(-1 != (opt = (getopt(argc, argv, "hvs:E:b:t:"))))
    {
        switch(opt)
        {
            case 'h':help_mode = 1;
                     break;
            case 'v':verbose_mode = 1;
                     break;
            case 's':s = atoi(optarg);
                     break;
            case 'E':E = atoi(optarg);
                     break;
            case 'b':b = atoi(optarg);
                     break;
            case 't':strcpy(filename, optarg);
                     break;
        }
    }
    if(help_mode == 1)
    {
        system("cat help_info");
        exit(0);
    }
    FILE* fp = fopen(filename,"r");
    if(fp == NULL)
    {
        fprintf(stderr,"The File is wrong!\n");
        exit(-1);
    }
    S = (1 << s); 
    cache = (cache_line**)malloc(sizeof(cache_line*) * S);
    for(int i = 0; i < S; i++)
        cache[i] = (cache_line*)malloc(sizeof(cache_line) * E);
    for(int i = 0; i < S; i++)
        for(int j = 0; j < E; j++)
        {
            cache[i][j].valid_bit = 0;
            cache[i][j].tag = cache[i][j].stamp = -1;
        }
    while(fgets(buffer,1000,fp))
    {
        sscanf(buffer," %c %xu,%d", &type, &address, &temp);
        switch(type)
        {
            case 'L':update(address);
                     break;
            case 'M':update(address);
            case 'S':update(address);
                     break;
        }
        update_time();
    }
    for(int i = 0; i < S; i++)
        free(cache[i]);
    free(cache);
    fclose(fp);
    printSummary(number_hits, number_miss, number_eviction);
    return 0;
}