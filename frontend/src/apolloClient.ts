import { ApolloClient, HttpLink, InMemoryCache } from '@apollo/client'

const httpLink = new HttpLink({
  uri: 'http://localhost:8001/graphql',
})

export const client = new ApolloClient({
  link: httpLink,
  cache: new InMemoryCache(),
})
