#include <iostream>
#include <iomanip>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
using namespace std;
const string MAP_FILE_PATH="../map/generated_map.txt";
const string OUTPUT_FILE_PATH="../map/generated_map.txt";
int xh,yh;
int xm,ym;
bool mp[30][30];//true if unavailable, false if available, read from file
int dis[30][30];
struct Node
{
    int x;
    int y;
    int val;
};
queue<Node> q;
int wrap(int v)
{
    return (v+30)%30;
}

int main()
{
    ifstream fin(MAP_FILE_PATH);
    if(!fin)
    {
        cerr<<"Cannot open map file. Please set MAP_FILE_PATH in bfs_escape.cpp."<<endl;
        return 1;
    }

    int cell;
    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            fin>>cell;
            mp[i][j]=(cell!=0);
        }
    }

    fin>>xh>>yh;//human
    fin>>xm>>ym;//monster

    Node tmp;
    tmp.x=xm;
    tmp.y=ym;
    tmp.val=1;

    q.push(tmp);

    while(!q.empty())
    {
        tmp=q.front();
        q.pop();

        if(mp[tmp.x][tmp.y]||dis[tmp.x][tmp.y])continue;

        dis[tmp.x][tmp.y]=tmp.val;
        q.push({wrap(tmp.x+1),tmp.y,tmp.val+1});
        q.push({tmp.x,wrap(tmp.y+1),tmp.val+1});
        q.push({wrap(tmp.x-1),tmp.y,tmp.val+1});
        q.push({tmp.x,wrap(tmp.y-1),tmp.val+1});
    }
/*
    for(int i=0;i<=29;i++)
    {
        for(int j=0;j<=29;j++)
        {
            if(i==xh&&j==yh)cout<<setw(3)<<"H";
            else if(i==xm&&j==ym)cout<<setw(3)<<"M";
            else if(dis[i][j]==0)cout<<setw(3)<<"#";
            else cout<<setw(3)<<dis[i][j];
        }
        cout<<endl;
    }
*/
    int down=dis[wrap(xh+1)][yh];
    int up=dis[wrap(xh-1)][yh];
    int right=dis[xh][wrap(yh+1)];
    int left=dis[xh][wrap(yh-1)];

    ofstream fout(OUTPUT_FILE_PATH);
    if(!fout)
    {
        cerr<<"Cannot open output file. Please set OUTPUT_FILE_PATH in bfs_escape.cpp."<<endl;
        return 1;
    }

    int d_max=max(down,max(up,max(right,left)));
    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            fout<<mp[i][j]<<" ";
        }
        fout<<endl;
    }
    if(down==d_max)fout<<wrap(xh+1)<<" "<<yh<<endl;
    else if(up==d_max)fout<<wrap(xh-1)<<" "<<yh<<endl;
    else if(right==d_max)fout<<xh<<" "<<wrap(yh+1)<<endl;
    else fout<<xh<<" "<<wrap(yh-1)<<endl;

    fout<<xm<<" "<<ym;
    return 0;
}
