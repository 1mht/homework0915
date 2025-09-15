## PART.1 data ->  json 

[{
    "id": "1", -- key
    "name": "xx",
    "sex": 1,
    "room_id": 101,
    "tele": 12312341234
 },
 {

 }
]


## PART.2 select, write, read(/all)




My require: 
1. read json
2. 3 choices
    2. select
    3. write
        1. 验证 (id(重复), tele(重复))
    4. read (2 steps)
3. code limited less than 200 lines