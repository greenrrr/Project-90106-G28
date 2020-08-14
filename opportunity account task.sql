

SELECT A.*,B.* from Opportunities AS A left join 
			(SELECT "WhatId" , MIN("ActivityDate") FROM "Tasks" GROUP BY "WhatId") AS B
on id = B.WhatId


SELECT A.*,B.* from Opportunities AS A left join
	(SELECT *,0 AS UoM_Organisation_Level__c FROM Account 
			UNION
	 SELECT * FROM "Account_internal"
	 ) AS B 
on AccountId = B.Id