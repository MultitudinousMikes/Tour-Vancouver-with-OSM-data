import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import lit
import pandas as pd
import numpy as np


def main(in_directory, out_directory):
    data = spark.read.load(in_directory, format='json',sep = ',')

    groupData = data.groupBy('amenity')
    resultData = groupData.agg(
        functions.count(data['amenity']).alias('amenityCount')
    )
    resultData.show();
    marketplace = data.filter(data.amenity == 'marketplace')
    arts_centre = data.filter(data.amenity == 'arts_centre')
    college = data.filter(data.amenity == 'college')
    university = data.filter(data.amenity == 'university')
    library = data.filter(data.amenity == 'library')
    bistro = data.filter(data.amenity == 'bistro')
    hookah_lounge = data.filter(data.amenity == 'hookah_lounge')
    nightClub = data.filter(data.amenity == 'nightclub')
    pub = data.filter(data.amenity == 'pub')
    social_centre = data.filter(data.amenity == 'social_centre')
    bar = data.filter(data.amenity == 'bar')
    casino = data.filter(data.amenity == 'casino')
    community_centre = data.filter(data.amenity == 'community_centre')
    theatre = data.filter(data.amenity == 'theatre')
    cinema = data.filter(data.amenity == 'cinema')
    bicycle_rental = data.filter(data.amenity == 'bicycle_rental')
    car_sharing = data.filter(data.amenity == 'car_sharing')
    boat_rental = data.filter(data.amenity == 'boat_rental')
    
    marketplace = marketplace.drop("tags","timestamp")
    market = marketplace.toPandas()
    market.to_csv('marketplace.csv',index=False)
    education = arts_centre.unionByName(college)
    education = education.unionByName(university)
    education = education.unionByName(library)
    education = education.drop("tags","timestamp")
    edu = education.toPandas()
    edu.to_csv('education.csv',index=False)
    
    bar = bar.unionByName(bistro)
    bar = bar.unionByName(nightClub)
    bar = bar.unionByName(pub)
    bar = bar.unionByName(hookah_lounge)
    bar = bar.drop("tags","timestamp")
    barPD = bar.toPandas()
    barPD.to_csv('bar.csv',index=False)
    
    casino = casino.drop("tags","timestamp")
    casinoPD = casino.toPandas()
    casinoPD.to_csv('casino.csv',index=False)
    
    activity = social_centre.unionByName(community_centre)
    activity = activity.drop("tags","timestamp")
    activityPD = activity.toPandas()
    activityPD.to_csv('activity.csv',index=False)
    
    film = theatre.unionByName(cinema)
    film = film.drop("tags","timestamp")
    filmPD = film.toPandas()
    filmPD.to_csv('film.csv',index=False)
    
    rental = bicycle_rental.unionByName(car_sharing)
    rental = rental.unionByName(boat_rental)
    rental = rental.drop("tags","timestamp")
    rentalPD = rental.toPandas()
    rentalPD.to_csv('rental.csv',index=False)
    
    
if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    spark = SparkSession.builder.appName('scenery').config("spark.sql.caseSensitive", "true").master('local[0]').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
