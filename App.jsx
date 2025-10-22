import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { Shield, MapPin, Phone, AlertTriangle, Users, Activity } from 'lucide-react'
import './App.css'

function App() {
  const [location, setLocation] = useState({ lat: null, lng: null })
  const [threatLevel, setThreatLevel] = useState('low')
  const [isTracking, setIsTracking] = useState(false)
  const [emergencyContacts, setEmergencyContacts] = useState([
    { name: 'Emergency Services', phone: '112' },
    { name: 'Family Contact', phone: '+91-9876543210' }
  ])
  const [alerts, setAlerts] = useState([])

  // Simulate location tracking
  useEffect(() => {
    if (isTracking) {
      const interval = setInterval(() => {
        // Simulate location updates
        setLocation({
          lat: 28.6139 + (Math.random() - 0.5) * 0.01,
          lng: 77.2090 + (Math.random() - 0.5) * 0.01
        })
        
        // Simulate threat level changes
        const levels = ['low', 'medium', 'high']
        setThreatLevel(levels[Math.floor(Math.random() * levels.length)])
      }, 3000)
      
      return () => clearInterval(interval)
    }
  }, [isTracking])

  const handleSOSAlert = () => {
    const newAlert = {
      id: Date.now(),
      type: 'SOS',
      timestamp: new Date().toLocaleTimeString(),
      location: location
    }
    setAlerts(prev => [newAlert, ...prev.slice(0, 4)])
    
    // Simulate sending alerts to emergency contacts
    alert('SOS Alert sent to emergency contacts and authorities!')
  }

  const getThreatColor = (level) => {
    switch (level) {
      case 'high': return 'bg-red-500'
      case 'medium': return 'bg-yellow-500'
      default: return 'bg-green-500'
    }
  }

  const getThreatBadgeVariant = (level) => {
    switch (level) {
      case 'high': return 'destructive'
      case 'medium': return 'secondary'
      default: return 'default'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Shield className="h-8 w-8 text-purple-600" />
            <h1 className="text-4xl font-bold text-gray-800">SafeGuard</h1>
          </div>
          <p className="text-lg text-gray-600">AI-Powered Women's Safety & Empowerment System</p>
        </div>

        {/* Main Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* SOS Panel */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-red-500" />
                Emergency SOS
              </CardTitle>
              <CardDescription>
                Press the button below in case of emergency
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={handleSOSAlert}
                className="w-full h-16 text-xl font-bold bg-red-500 hover:bg-red-600"
                size="lg"
              >
                🚨 SOS ALERT
              </Button>
              <div className="text-sm text-gray-600">
                This will immediately alert your emergency contacts and nearby authorities
              </div>
            </CardContent>
          </Card>

          {/* Location & Threat Status */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="h-5 w-5 text-blue-500" />
                Location & Safety Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span>Location Tracking:</span>
                <Button 
                  onClick={() => setIsTracking(!isTracking)}
                  variant={isTracking ? "default" : "outline"}
                  size="sm"
                >
                  {isTracking ? 'ON' : 'OFF'}
                </Button>
              </div>
              
              {location.lat && (
                <div className="text-sm text-gray-600">
                  <div>Lat: {location.lat.toFixed(4)}</div>
                  <div>Lng: {location.lng.toFixed(4)}</div>
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <span>Threat Level:</span>
                <Badge variant={getThreatBadgeVariant(threatLevel)}>
                  {threatLevel.toUpperCase()}
                </Badge>
              </div>
              
              <div className={`h-2 rounded-full ${getThreatColor(threatLevel)}`}></div>
            </CardContent>
          </Card>

          {/* Emergency Contacts */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Phone className="h-5 w-5 text-green-500" />
                Emergency Contacts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {emergencyContacts.map((contact, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div>
                    <div className="font-medium">{contact.name}</div>
                    <div className="text-sm text-gray-600">{contact.phone}</div>
                  </div>
                  <Button size="sm" variant="outline">
                    Call
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="text-center p-6">
            <Activity className="h-12 w-12 text-purple-500 mx-auto mb-4" />
            <h3 className="font-semibold mb-2">Real-time Monitoring</h3>
            <p className="text-sm text-gray-600">AI continuously monitors your safety status</p>
          </Card>
          
          <Card className="text-center p-6">
            <MapPin className="h-12 w-12 text-blue-500 mx-auto mb-4" />
            <h3 className="font-semibold mb-2">Safe Route Suggestions</h3>
            <p className="text-sm text-gray-600">Get the safest routes to your destination</p>
          </Card>
          
          <Card className="text-center p-6">
            <Users className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="font-semibold mb-2">Community Support</h3>
            <p className="text-sm text-gray-600">Connect with other women in your area</p>
          </Card>
          
          <Card className="text-center p-6">
            <Shield className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="font-semibold mb-2">Predictive Alerts</h3>
            <p className="text-sm text-gray-600">AI predicts and warns about potential threats</p>
          </Card>
        </div>

        {/* Recent Alerts */}
        {alerts.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {alerts.map((alert) => (
                  <Alert key={alert.id}>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      {alert.type} Alert sent at {alert.timestamp}
                      {alert.location.lat && ` - Location: ${alert.location.lat.toFixed(4)}, ${alert.location.lng.toFixed(4)}`}
                    </AlertDescription>
                  </Alert>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export default App

