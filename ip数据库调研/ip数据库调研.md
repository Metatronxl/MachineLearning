###  1) ipip.net

>
>- 基于BGP/ASN数据分析处理而得来的IP库
- 提供 IP 所属 AS，IP 所属上游，AS 名称及其所属上下游关系等四项数据
- 针对全网 IPv4 进行 30 多个常用端口的扫描与协议（HTTP,HTTPS,SSH,VPN,SOCKS4,SOCKS5,HTTP_PROXY等等）



### 2) RTBAsia
>
>- 提供非人类流量监控API
- 包含四个API，分别为IP地址经纬度覆盖面，IP地址类型查询,IP地址真人度查询,IP地址区县级归属地


### 3) MaxMind GeoIP
>- Country，City，ISP，DomainName，ConnectionType



### 4) ZoomEye
> 获取网络指纹
>
- 网站指纹包括应用名、版本、前端框架、后端框架、服务端语言、服务器操作系统、网站容器、内容管理系统和数据库等。设备指纹包括应用名、版本、开放端口、操作系统、服务名、地理位置等

- 其中根据IP可以查出开放的端口、服务、国家、年份、组件


- 根据CIDR IP端进行查询

![](/Users/xulei2/Desktop/ip数据库调研/屏幕快照 2018-01-06 下午4.13.11.png)

### 5） Shodan
> 提供国家、组织、ISP、HostName、ASN、最后更新日期


![](/Users/xulei2/Desktop/ip数据库调研/屏幕快照 2018-01-06 下午4.20.12.png)

### 6) censys
> 提供Whois信息查询

![](/Users/xulei2/Desktop/ip数据库调研/屏幕快照 2018-01-06 下午4.25.56.png)

### 7） whoisxmlapi
> 提供whois信息查询

- 基于American Registry for Internet Numbers (ARIN)

![](/Users/xulei2/Desktop/ip数据库调研/屏幕快照 2018-01-06 下午4.48.32.png)

### 8）domainTools
> 
- 1.域名搜索：搜索相匹配的活跃着的或者已经被删除的域名
- 2.IP监控： 对给予的IP地址提供更新或者删除域名的通知
- 3.Reverse IP Whois：根据搜索信息在whois数据中匹配对应的IP地址
- 4.whois信息查询

>  全部API产品介绍地址：http://domaintools.com/resources/api-documentation/

####注：未在文档中记录的网站为与whois信息无关或者限制打开

