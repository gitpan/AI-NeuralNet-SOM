package AI::NeuralNet::SOM;

use strict;
use warnings;

require Exporter;
use base qw(Exporter);

use Data::Dumper;

=pod

=head1 NAME

AI::NeuralNet::SOM - Perl extension for Kohonen Maps

=head1 SYNOPSIS

  use AI::NeuralNet::SOM;
  my $nn = new AI::NeuralNet::SOM::Rect (output_dim => "5x6",
                                         input_dim  => 3);
  $nn->initialize;
  $nn->train (30, 
    [ 3, 2, 4 ], 
    [ -1, -1, -1 ],
    [ 0, 4, -3]);

  print $nn->as_data;

  my $nn = new AI::NeuralNet::SOM::Hexa (output_dim => 6,
                                         input_dim  => 4);
  $nn->initialize ( [ 0, 0, 0, 0 ] );  # all get this value
  $nn->value (3, 2, [ 1, 1, 1, 1 ]);   # change value for a neuron

=head1 DESCRIPTION

This package is a stripped down implementation of the Kohonen Maps
(self organizing maps). It is B<NOT> meant as demonstration or for use
together with some visualisation software. And while it is not (yet)
optimized for speed, some consideration has been given that it is not
overly slow.

Particular emphasis has been given that the package plays nicely with
others. So no use of files, no arcane dependencies, etc.

=head2 Scenario

The basic idea is that the neural network consists of a 2-dimensional
array of N-dimensional vectors. When the training is started these
vectors may be complete random, but over time the network learns from
the sample data, also N-dimensional vectors.

Slowly, the vectors in the network will try to approximate the sample
vectors fed in. If in the sample vectors there were clusters, then
these clusters will be neighbourhoods within the rectangle.

Technically, you have reduced your dimension from N to 2.

=head1 INTERFACE

=head2 Constructor

The constructor takes arguments:

=over

=item C<input_dim> : (mandatory, no default)

A positive integer specifying the dimension of the sample vectors (and hence that of the vectors in
the grid).

=item C<learning_rate>: (optional, default C<0.1>)

This is a magic number which influence how strongly the vectors in the grid can be
influenced. Higher movement can mean faster learning if the clusters are very pronounced. If not,
then the movement is like noise and the convergence is not good. To mediate that effect, the
learning rate is reduced over the iterations.

=item C<sigma0>: (optional, defaults to radius)

A non-negative number representing the start value for the learning radius. It starts big, but then
narrows as learning time progresses. This makes sure that the network finally has only localized
changes.

=back

Subclasses will (re)define some of these parameters and add others:

Example:

    my $nn = new AI::NeuralNet::SOM::Rect (output_dim => "5x6",
				           input_dim  => 3);

=cut

sub new { die; }

=pod

=head2 Methods

=over

=item I<initialize>

I<$nn>->initialize

You need to initialize all vectors in the map before training. There are several options
how this is done:

=over

=item providing data vectors

If you provide a list of vectors, these will be used in turn to seed the neurons. If the list is
shorter than the number of neurons, the list will be started over. That way it is trivial to
zero everything:

  $nn->initialize ( [ 0, 0, 0 ] );

=item providing no data

Then all vectors will get randomized values (in the range [ -0.5 .. 0.5 ]).

=back

TODO: Eigenvectors

=item I<train>

I<$nn>->train ( I<$epochs>, I<@vectors> )

The training uses the list of sample vectors to make the network learn. Each vector is simply a
reference to an array of values. Individual vectors are

The C<epoch> parameter controls how many vectors are processed.

Example:

   $nn->train (30, 
               [ 3, 2, 4 ],
               [ -1, -1, -1 ], 
               [ 0, 4, -3]);

TODO: @@@@@@ expose error

=cut

sub train {
    my $self   = shift;
    my $epochs = shift || 1;

    $self->{LAMBDA} = $epochs / log ($self->{_Sigma0});                                 # educated guess?

    for my $epoch (1..$epochs){
	$self->{T} = $epoch;
	my $sigma = $self->{_Sigma0} * exp ( - $self->{T} / $self->{LAMBDA} );          # compute current radius
	my $l     = $self->{_L0}     * exp ( - $self->{T} / $epochs );                  # current learning rate

	my $sample = @_ [ int (rand (scalar @_) ) ];                                    # take random sample

	my @bmu = $self->bmu ($sample);                                                 # find the best matching unit
#warn "bmu ".Dumper \@bmu;
	my $neighbors = $self->neighbors ($sigma, @bmu);                                # find its neighbors
#warn "neighbors ".Dumper $neighbors;
	map { _adjust ($self, $l, $sigma, $_, $sample) } @$neighbors;                   # bend them like Beckham
#warn Dumper $self;
    }
}

sub _adjust {                                                                           # http://www.ai-junkie.com/ann/som/som4.html
    my $self  = shift;
    my $l     = shift;                                                                  # the learning rate
    my $sigma = shift;                                                                  # the current radius
    my $unit  = shift;                                                                  # which unit to change
    my ($x, $y, $d) = @$unit;                                                           # it contains the distance
    my $v     = shift;                                                                  # the vector which makes the impact

    my $w     = $self->{map}->[$x]->[$y];                                               # find the data behind the unit
    my $theta = exp ( - ($d ** 2) / (2 * $sigma ** 2));                                 # gaussian impact (using distance and current radius)

    foreach my $i (0 .. $#$w) {                                                         # adjusting values
	$w->[$i] = $w->[$i] + $theta * $l * ( $v->[$i] - $w->[$i] );
    }
}

=pod

=item I<bmu>

(I<$x>, I<$y>, I<$distance>) = I<$nn>->bmu (I<$vector>)

This method finds the I<best matching unit>, i.e. that neuron which is closest to the vector passed
in. The method returns the coordinates and the actual distance.

=cut

sub bmu { die; }

=pod

=item I<mse>

I<$mse> = I<$nn>->mse (I<@vectors>)

=cut

=pod

=item I<neighbors>

I<$ns> = I<$nn>->neighbors (I<$sigma>, I<$x>, I<$y>)

Finds all neighbors of (X, Y) with a distance smaller than SIGMA. Returns a list reference of (X, Y,
distance) triples.

=cut

sub neighbors { die; }

=pod

=item I<output_dim> (read-only)

I<$dim> = I<$nn>->output_dim

Returns the output dimensions of the map as passed in at constructor time.

=cut

sub output_dim {
    my $self = shift;
    return $self->{output_dim};
}

=pod

=item I<radius> (read-only)

I<$radius> = I<$nn>->radius

Returns the I<radius> of the map. Different topologies interpret this differently.

=item I<map>

I<$m> = I<$nn>->map

This method returns a reference to the map data. See the appropriate subclass of the data
representation.

=cut

sub map {
    my $self = shift;
    return $self->{map};
}

=pod

=item I<value>

Set or get the current vector value for a particular neuron. The neuron is addressed via its
coordinates.

=cut

sub value {
    my $self    = shift;
    my ($x, $y) = (shift, shift);
    my $v       = shift;
    return $v ? $self->{map}->[$x]->[$y] = $v : $self->{map}->[$x]->[$y];
}

=pod

=item I<as_string>

print I<$nn>->as_string

This methods creates a pretty-print version of the current vectors.

=cut

sub as_string { die; }

=pod

=item I<as_data>

print I<$nn>->as_data

This methods creates a string containing the raw vector data, row by
row. This can be fed into gnuplot, for instance.

=cut

sub as_data { die; }

=pod

=back


=head1 SEE ALSO

L<http://www.ai-junkie.com/ann/som/som1.html>

=head1 AUTHOR

Robert Barta, E<lt>rho@devc.atE<gt>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2007 by Robert Barta

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.8 or,
at your option, any later version of Perl 5 you may have available.

=cut

our $VERSION = '0.03';

1;

__END__


sub bmu {
    my $self = shift;
    my $sample = shift;

    my $closest;                                                               # [x,y, distance] value and co-ords of closest match
    foreach my $coor ($self->_get_coordinates) {                               # generate all coord pairs, not overly happy with that
	my ($x, $y) = @$coor;
	my $distance = _vector_distance ($self->{map}->[$x]->[$y], $sample);   # || Vi - Sample ||
	$closest = [0,  0,  $distance] unless $closest;
	$closest = [$x, $y, $distance] if $distance < $closest->[2];
    }
    return @$closest;
}

