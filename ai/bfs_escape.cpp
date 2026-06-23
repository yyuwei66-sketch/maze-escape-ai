#include <iostream>
#include <iomanip>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <random>
using namespace std;
const string MAP_FILE_PATH="../map/generated_map.txt";
const string OUTPUT_FILE_PATH="../map/generated_map.txt";
int xh,yh;
int xm,ym;
int previous_xh=-1,previous_yh=-1;
bool has_previous_human=false;
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
    int previousHumanFlag=0;
    if(fin>>previousHumanFlag)
    {
        if(previousHumanFlag==2&&fin>>previous_xh>>previous_yh)
        {
            previous_xh=wrap(previous_xh);
            previous_yh=wrap(previous_yh);
            has_previous_human=true;
        }
    }

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

    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            fout<<mp[i][j]<<" ";
        }
        fout<<endl;
    }

    vector<pair<int,pair<int,int>>> candidates={
        {down,{wrap(xh+1),yh}},
        {up,{wrap(xh-1),yh}},
        {right,{xh,wrap(yh+1)}},
        {left,{xh,wrap(yh-1)}}
    };

    vector<pair<int,pair<int,int>>> legalMoves;
    vector<pair<int,pair<int,int>>> forwardMoves;
    for(const auto& candidate:candidates)
    {
        const auto& pos=candidate.second;
        if(mp[pos.first][pos.second])continue;
        legalMoves.push_back(candidate);
        if(!has_previous_human||pos.first!=previous_xh||pos.second!=previous_yh)
        {
            forwardMoves.push_back(candidate);
        }
    }

    // Avoid immediately backtracking when another legal move exists. In a
    // dead end, however, returning to the previous cell is the only way out.
    const auto& availableMoves=forwardMoves.empty()?legalMoves:forwardMoves;
    vector<pair<int,int>> bestMoves;
    int d_max=-1;
    for(const auto& candidate:availableMoves)
    {
        if(candidate.first>d_max)d_max=candidate.first;
    }

    for(const auto& candidate:availableMoves)
    {
        if(candidate.first==d_max)bestMoves.push_back(candidate.second);
    }

    pair<int,int> nextHuman={xh,yh};
    if(!bestMoves.empty())
    {
        random_device rd;
        mt19937 rng(rd());
        uniform_int_distribution<int> pick(0,(int)bestMoves.size()-1);
        nextHuman=bestMoves[pick(rng)];
    }
    fout<<nextHuman.first<<" "<<nextHuman.second<<endl;

    fout<<xm<<" "<<ym;
    return 0;
}
