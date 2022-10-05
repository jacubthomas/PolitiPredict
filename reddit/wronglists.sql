USE Reddit;
SELECT * FROM WrongLists;

SELECT * FROM WrongLists WHERE id = '11j8k5';

DROP TABLE WrongLists;

CREATE TABLE WrongLists (
id VARCHAR (15) NOT NULL,
posts VARCHAR(500) NOT NULL,
party VARCHAR(25) NOT NULL,
count INT NOT NULL DEFAULT (0),
PRIMARY KEY (id)
);

SELECT * FROM WrongLists
ORDER BY count DESC;

SELECT * FROM WrongLists
ORDER BY count ASC;