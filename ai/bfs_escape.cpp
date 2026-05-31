#include <iostream>
#include <iomanip>
#include <queue>
#include <algorithm>
using namespace std;
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
int main()
{
    /*
    mp[11][11]=true;
    mp[11][12]=true;
    mp[11][13]=true;
    */

    cin>>xh>>yh;//human
    cin>>xm>>ym;//monster

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
        q.push({(30+tmp.x+1)%30,tmp.y,tmp.val+1});
        q.push({tmp.x,(30+tmp.y+1)%30,tmp.val+1});
        q.push({(30+tmp.x-1)%30,tmp.y,tmp.val+1});
        q.push({tmp.x,(30+tmp.y-1)%30,tmp.val+1});
    }
/*
    for(int i=0;i<=29;i++)
    {
        for(int j=0;j<=29;j++)
        {
            cout<<setw(3)<<dis[i][j];
        }
        cout<<endl;
    }
*/
    int d_max=max(dis[xh+1][yh],max(dis[xh-1][yh],max(dis[xh][yh+1],dis[xh][yh-1])));
    if(dis[xh+1][yh]==d_max)cout<<1<<" "<<0<<endl;
    else if(dis[xh-1][yh]==d_max)cout<<-1<<" "<<0<<endl;
    else if(dis[xh][yh+1]==d_max)cout<<0<<" "<<1<<endl;
    else cout<<0<<" "<<-1<<endl;

    return 0;
}