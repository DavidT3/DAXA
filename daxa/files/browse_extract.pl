#!/usr/bin/perl
#=========================================================================#
# High Energy Astrophysics Science Archive Research Center (HEASARC),
# NASA/Goddard Space Flight Center, Greenbelt, MD 20771, USA.
#
# You are free to use this code as you wish, but unless you change
# it significantly, please leave the above name and address intact.
#=========================================================================#
#
# This program creates returns a table giving the results of queries
# of the HEASARC database.  Users may enter a single object
# or may extract information for a list of objects.
#
# Usage:
#   browse_extract.pl table=table
#                     [position=name|coordinates|null]
#                     [time=time|time_range]
#                     [coordinates=Equatorial|Galactic]
#                     [equinox=year]
#                     [radius=arcmin]
#                     [fields=Standard|All|list]
#                     [name_resolver=CheckCaches/GRB/SIMBAD/NED|CheckCaches/NED/SIMBAD|NED|SIMBAD]
#                     [infile=input_list]
#                     [outfile=output_file]
#                     [format=Batch|FITS|VOTable|Excel|HTML|Text]
#                     [sortvar=column_name]
#                     [resultmax=count]
#                     [param=name,value  /or/ name=value]...
#   or
#
#   browse_extract.pl table=xxx
#      to just get a list of available tables.  Only VizieR tables
#      directly linked within the HEASARC will be noted, but any
#      VizieR table can be queried using browse_extract.
#
#  Table is the Table name as used in the HEASARC, e.g., abell, rospublic
#    This is given as the short name in the Browse web interfaces.
#
#  Position is either the name of an object or a set of coordinates
#    which giving the center of the search.  If a name is given it will
#    be resolved using the service give in the name_resolver keyword
#    or SIMBAD by default.  The syntax for coordinates is that supported
#    in the Browse web services.  If the coordinates string contains
#    embedded spaces, then this argument should be enclosed in quotes.
#    Multiple positions may be separated by semicolons but these will then be
#    processed together giving a combined count for all specified targets.
#
#    The Position argument can be specified as "none" or "null" if the user
#    does not wish to specify a position (e.g., to query on exposure only).
#
#  Time allows the user to request a temporal query on the table if a time
#    column has been defined.  This acts in the same fashion as the Time query
#    column in the web version of Browse.  Times may be specified in ISO or
#    MJD formats.  (This is also true when doing a parameter search of a time
#    column as well.)  Multiple times or intervals can be specified by
#    separating them with semicolons.
#
#  The coordinates argument should be either "Equatorial" or "Galactic".
#    The default is Equatorial.  It is used to determine the input coordinates
#    if used and the display coordinate system for the primary coordinates in
#    the table.
#
#  Equinox gives the equinox year for input and output coordinates.  It
#    defaults to 2000 (and is ignored for Galactic coordinates).
#
#  Radius gives the radius in arcminutes out to which the search is to take
#    place.  This defaults to 60.
#
#  The Fields argument indicates which parameters are to be retrieved
#    from the table.  The default, "Standard", indicates that only
#    a limited set of parameters will be retrieved.   "All" will retrieve
#    all parameters from the table.  A list of specific fields separated
#    by commas may also be specified.  Some system generated columns
#    may be included even when a specific list is requested.
#
#  The Name_Resolver options currently supported are NED and SIMBAD.  The
#    default is CheckCaches/SIMBAD/NED which first checks the HEASARC
#    caches and then tries the other two services in the order specified.
#
#  The Param=name,value argument is used to do parameter searches.
#    The syntax of the value parameter is the same as used in the Browse web
#    interface, e.g., 3000, >5000, 4..10, 3C*273 are possible values which
#    search for data with a value equal to 3000, greater than 5000, between 4
#    and 10 or matching the strings '3C273', '3CXXXXX273', etc.,
#    respectively.  If the name of the parameter does not conflict with one of
#    the other arguments to browse_extract, then the simpler syntax
#        name=value
#    may be used.
#
#    In some environments characters in the value may need to be escaped, e.g.,
#        exposure='>2000' or exposure=\>2000
#
#  The Infile argument specifies a file containing objects to be
#    searched.  Each line in the file will be used as the
#    Position.  If no Infile or Position argument is give then
#    the positions will be read from the standard input.
#
#  The Outfile argument specifies a file to contain the table of
#    returned results.
#
#  Use resultmax to specify the maximum number of rows to be returned.
#    Use 0 to specify no limit.  The default is 250.
#
#  The Format argument specifies the desired output format. When anything
#    other than the batch format is requested, all positions will be searched
#    at once. In batch queries each line of the input is specified as a
#    distinct query.
#    Current valid formats include:
#      Batch   - The default format.  Each position in the infile or stdin is
#                processed separately.
#      HTML    - The Text Table format in the web version of Browse.
#      FITS    - A FITS table. (The results are in the first extension.)
#      EXCEL   - An Excel compatible output format.
#      VOTable - The Virtual Observatory Table format.
#      Text    - The Pure Text format in the web version of Browse.
#
#   The Sortvar may be used to specify the field on which the results will be
#     sorted. This variable need not be displayed.
#
#  All argument keys are case-insensitive except those for VizieR tables.
#
#  Other reserved keywords: host, url
#
# EndUsage
#
# Browse extract runs as a pre and post processor to the simple CLI web
# extraction tool to a CLI browser.
#
######################################################################
#
# Version 1.0           1996-10-10 by T. McGlynn    NASA/GSFC/USRA
# Version 2.0           1998-02-26 by Saima Zobair  Raytheon STX
# Version 2.1           2005-05-31 by T. McGlynn    NASA/GSFC
# Version 2.1a          2006-03-28 by T. McGlynn    Made formats case-
#                                                   insensitive.
# Version 2.1b          2006-09-22 by T. McGlynn    Use WGET rather than
#                                                   webquery.
# Version 2.2           2010-09-07 by  L. McDonald  Parameters are no longer
#                                                   converted to lower case
#                                                   for VizieR tables.
#                                                   Parameter syntax now matches
#                                                   browse_extract.pl
# Version 2.3           2021-08-16 by  E. Sabol     Adapted to use curl (if
#                                                   present). Minor changes.
######################################################################

use strict;

use constant DEBUG => 0;

# Determine which retrieval program to use from the name of the executable.
my $which = ($0 =~ /curl/) ? "curl" : ($0 =~ /wget/) ? "wget" : "";

# $which = "curl";    # Uncomment this line to use curl explicitly.
# $which = "wget";    # Uncomment this line to use wget explicitly.

my $retrieve = (($which eq "curl" || $which eq "") && (-x "/usr/bin/curl"))
    ? "curl -o - --silent -k"
    : "wget -O - -o /dev/null --no-check-certificate";

my $host     = "heasarc.gsfc.nasa.gov";
my $batch    = "BATCHRETRIEVALCATALOG_2.0";

if ($#ARGV < 0) {
    $which = "_" . $which;
    print <<EOT;

Usage:
  browse_extract${which}.pl table=table
                     [position=name|position|null]
                     [time=iso_or_mjd_time|time_range]
                     [coordinates=Equatorial|Galactic]
                     [equinox=year]
                     [radius=arcmin]
                     [fields=Standard|All|list]
                     [name_resolver=CheckCaches/GRB/SIMBAD/NED|CheckCaches/NED/SIMBAD|NED|SIMBAD]
                     [infile=input_list]
                     [outfile=output_file]
                     [format=Batch|FITS|VOTable|Excel|HTML|Text]
                     [sortvar=column_name]
                     [resultmax=count]
                     [param=name,value  /or/ name=value]...

            .. or ..

  browse_extract${which}.pl table=xxx

  The simpler form for the parameter queries may be used so long as the
  name does not conflict with another browse_extract${which}.pl parameter.
  The second form of the command simply prints out the tables available
  ("xxx" can be any string that is not a table name).
EOT

    exit;
}

#  Process arguments.
my %args;

my @stdargs =  ("table", "radius", "fields",
                "name_resolver", "coordinates",
                "equinox", "resultmax", "sortvar",
                "format", "infile", "outfile",
                "position", "url", "time",
               );

$args{radius}        = 60;
$args{fields}        = "standard";
#$args{name_resolver} = "CheckCaches/SIMBAD/NED";
$args{coordinates}   = "equatorial";
$args{equinox}       = 2000;
$args{resultmax}     = 250;
$args{format}        = "batch";

my %stdFormats = (
   fits    => "FitsDisplay",
   text    => "PureTextDisplay",
   excel   => "ExcelDisplay",
   votable => "VODisplay",
   batch   => "BatchDisplay",
   html    => "SimpleDisplay",
);

my $hasParam = 0;

# Process arguments.
my $vizier = 0;

foreach my $arg (@ARGV) {
    #--- look for VizieR table first (slightly different processing for VizieR)
    my ($key, $val) = split(/=/, $arg, 2);

    if ($key =~ /^\s*table\s*$/i) {
        if ($val =~ /\//) {
          $vizier = 1;
          last;
        }
    }
}


# Process arguments now that we have table information.
foreach my $arg (@ARGV) {

    my ($key, $val) = split(/=/, $arg, 2);

    if (! $vizier) {           # do not convert VizieR parameters to lc
        $key = lc($key);
    }

   if (!updateArgs(\%args, $key, $val, \@stdargs) ) {
       if ($key eq "param") {
           ($key, $val) = split(/,/, $val, 2);
       }
       updateParams(\%args, $key, $val);
   }
}

if (!defined($args{table})) {
   die "Error: A table parameter must be specified.\n";
}

if ($args{table} eq "xxx") {
   $args{position} = "none";
}

if (defined($args{position}) && defined($args{infile})) {
   die "Error: Please specify position[s] in only one of command line or input file\n";
}
my @objects;
if (defined($args{position})) {
   @objects = ($args{position});
}

# Set input and output.  Redefine STDIN and STDOUT if
# user has specified appropriate arguments.
if (defined ($args{infile})) {
   open (my $in_fh,'<', $args{infile}) || die "Error: Unable to open input file $args{infile}: $!\n";
   chomp(@objects = <$in_fh>);
   close($in_fh) || die "Error: Could not close input file $args{infile}: $!\n";
}

if (defined($args{outfile})) {
   close(STDOUT) || die "Error: Could not close STDOUT: $!\n";
   open(STDOUT, '>', $args{outfile}) || die "Error: Unable to open output file $args{outfile}: $!\n";
}

# If not using batch interface we need to
# concatenate positions first.
if ($args{format} ne "batch") {
   if (!(@objects)) {
       @objects = <STDIN>;
   }

   if (@objects > 1) {
       @objects = (join(";", @objects));
   }

   if (@objects < 1) {
       die "Error: Unable to read objects.\n";
   }
}

if (@objects > 0) {

   foreach my $obj (@objects) {
       $args{position} = $obj;
       do_position(%args);
   }

} else {

   if (!defined($args{infile})) {
       print "Please enter positions to be queried (EOF to terminate)\n";
   }
   while (my $obj = <STDIN>) {
       chomp($obj);
       $args{position} = $obj;
       do_position(%args);
   }

}

#------------------------------------------------------------------------
sub do_position {

   my (%args) = @_;

   # These define the URL to be used.
   my $url       = "/db-perl/W3Browse/w3query.pl";

   my $table     = lc($args{"table"});

   my $Tablehead = "name=$batch $table";
   my $Action    = "Query";

   my ($Entry)   = $args{position};

   my ($Coordinates);

   if ($args{coordinates} =~ /equatorial/i) {
       $Coordinates = "'Equatorial: R.A. Dec'";
   } elsif($args{coordinates} =~ /galactic/i) {
       $Coordinates = "'Galactic: LII BII'";
   } else {
       $Coordinates = $args{coordinates};
   }

   my ($Equinox) = $args{equinox};
   my ($Radius)  = $args{radius};

   my ($NR);

   if (defined($args{name_resolver})) {
       if ($args{name_resolver} =~ /^\s*simbad\s*$/i) {
           $NR = "SIMBAD";
       } elsif ($args{name_resolver} =~ /^\s*ned\s*$/i) {
           $NR = "NED";
       } else {
           $NR = $args{name_resolver};
       }
   } else {
       $NR = "CheckCaches/SIMBAD/NED";
   }

   my ($GIFsize) = 0;

   my ($Fields);

   if ($args{fields}      =~ /standard/i) {
       $Fields = "Standard";
   } elsif ($args{fields} =~ /all/i) {
       $Fields = "All";
   } else {
       my (@flds) = split(/,/, $args{fields});
       $Fields = "";
       foreach my $fld (@flds) {
           $Fields .= "&varon=" . webcode($fld);
       }
   }

   # Allow user to override default locations (but we don't advertise this).
   if ($args{url}) {
       $url = $args{url};
   }

   if ($args{host}) {
       $host = $args{host};
   }

   # The webquery.pl script is assumed to be in the path.
   my $cmd = "$retrieve 'https://$host$url?" .
                      "tablehead="    . webcode($Tablehead)   .
                      "&Action="      . webcode($Action)      .
                      "&Coordinates=" . webcode($Coordinates) .
                      "&Equinox="     . webcode($Equinox)     .
                      "&Radius="      . webcode($Radius)      .
                      "&NR="          . webcode($NR)          .
                      "&GIFsize="     . webcode($GIFsize)     .
                      "&Fields="      . $Fields;

   if (lc($Entry) ne "none"  and lc($Entry) ne "null") {
       $cmd .= "&Entry=" . webcode($Entry);
   } else {
       delete($args{position});
   }

   $cmd .= "&sortvar="    . webcode($args{sortvar})     if $args{sortvar};
   $cmd .= "&ResultMax="  . webcode($args{resultmax})   if defined($args{resultmax});
   $cmd .= "&Time="       . webcode($args{"time"})      if $args{"time"};

   if (defined($args{params})) {
       my $params = $args{params};
       $cmd .= "&" . join("&", @$params);
   }


   my $format;
   if (defined($args{format})) {
       $format = $args{format};
   } else {
       $format = "batch";
   }

   my $display = $stdFormats{lc($format)};
   if (!defined($display)) {
       $display = "BatchDisplay";
   }

   $cmd .= "&displaymode=" . webcode($display);


   # Close the single quote.
   $cmd .= "'";

   print "Command: $cmd\n" if DEBUG;
   my @results = `$cmd`;

   if ($format ne "batch") {
       print @results;
   } else {
       processResults(\@results, %args);
   }

   return;
}

# Batch processing of results.
sub processResults {

   my ($results, %args) = @_;

   checkError($$results[0]);

   my $count     = 0;
   my $seenstart = 0;
   my $error     = 0;

   for (@$results) {
       # Skip blanks.
       next if (/^\s*$/);

        if (!$seenstart) {
            if (/^BatchStart/) {
               $seenstart = 1;
            }
        } else {
            if (/^Content-type:/) {
                $seenstart = 0;
                next;
            }

            if ($count == 0) {
                checkError($_);
            }

            if (/does not seem to exist/) {
                $error = 1;
            }

           if (/^BatchEnd/) {
                last;
            }
            print $_;
            $count += 1;
        }
   }         #end for each line of results

   if (! $seenstart) { #checks if no results were displayed
       &checkNoResults(@$results);
   }

   if ($count > 0) {
       $count -= 2;
        if ($count < 0) {
            $count = 0;
        }
   }

   if (!$error) {
       print "Search of table $args{table} ";
       if (defined($args{position})) {
           print "around $args{position} with a radius $args{radius}' ";
       }
       if (defined($args{params})) {
           print "\n   ";
           foreach my $qual (@{$args{params}}) {
                if ($qual =~ /bparam_([^=]*)='(.*)'/) {
                    my $fld = $1;
                    my $val = $2;
                    if ($val !~ /<|>|=/) {
                        $val = "= $val";
                    }
                }
            }
        }

       my $s = "s";
       if ($count == 1) {
           $s = "";
       }
       print "returned $count row$s\n";
   }

   return;

} #eosub--do_position

#------------------------------------------------------------------------
sub checkError {

    # Check for the "BATCH_RETRIEVAL_MSG" string to detect error.
    my($check_result) = @_;

    if ($check_result =~ /^BATCH_RETRIEVAL_MSG\s*/) {
	print STDERR "$'";
	exit -1;
    }

    return;

} #eosub--checkError

#------------------------------------------------------------------------
sub checkNoResults {

    # Look for an informative message if no results were found.
    for (@_) {
	if (/\<h2\>\s*Error\s+(.*)\<\/h2\>/i) {
	    print STDERR "\nError: $1\n";
	    print STDERR "\nSQL errors may indicate use of invalid columns or invalid data types in parameter queries\n";
	    exit -1;
	} elsif (/\<h2\>Database\stemporarily\sunavailable/) {
	    print STDERR "\nUnable to connect to table: Incorrect table name or W3Browse not up.\n";
	    exit -1;
	}
    }

    return;

} #eosub--checkNoResults
#------------------------------------------------------------------------

# Process standard arguments
sub updateArgs {

   my ($args, $key, $val, $check) = @_;

   foreach my $fld (@$check) {
       if ($key eq $fld) {
           $$args{$key} = $val;
           return 1;
       }
   }
   return 0;
}
#------------------------------------------------------------------------

# Add a parameter query.
sub updateParams {

   my ($args, $key, $val) = @_;

   my $params = $$args{params};
   if (!defined($params)) {
       $params = [];
   }
   push(@$params, "bparam_$key=" . webcode($val));
   $$args{params} = $params;

   return;

}
#------------------------------------------------------------------------

# Imported from webquery.pl
sub webcode {

   my($string) = @_;

   # First convert special characters to to %xx notation.
   $string =~ s/[^ a-zA-Z0-9]/sprintf("%%%02x", ord($&))/eg ;

   # Now convert the spaces to +'s.
   # We do this last since otherwise +'s would be translated by th above.
   $string =~ tr/ /+/;

   # Perl doesn't require (or even like) putting in the return value,
   # but I find it clearer.
   return $string;
}
#------------------------------------------------------------------------