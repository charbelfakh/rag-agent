import { ApolloClient, HttpLink, InMemoryCache, split } from '@apollo/client'
import { GraphQLWsLink } from '@apollo/client/link/subscriptions'
import { getMainDefinition } from '@apollo/client/utilities'
import { createClient } from 'graphql-ws'

const HTTP_URI = 'http://localhost:8001/graphql'
const WS_URI = 'ws://localhost:8001/graphql'

const httpLink = new HttpLink({ uri: HTTP_URI })

const wsLink = new GraphQLWsLink(createClient({ url: WS_URI }))

// Route subscriptions over the websocket transport, everything else over HTTP.
const link = split(
  ({ query }) => {
    const definition = getMainDefinition(query)
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    )
  },
  wsLink,
  httpLink,
)

export const client = new ApolloClient({
  link,
  cache: new InMemoryCache(),
})
